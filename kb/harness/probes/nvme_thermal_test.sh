#!/usr/bin/env bash
# NVMe sustained-read bandwidth + thermal throttling diagnostic.
# Read-only (direct I/O) - safe on disks with mounted filesystems.
#
# Usage:
#   ./nvme_thermal_test.sh [runtime_sec] [/dev/nvmeXnY ...]
# Defaults: 60 s per disk, all NVMe devices found in the system.
# Re-execs itself with sudo automatically.
#
# What it does per disk:
#   1. SMART snapshot (temp, throttle counters)
#   2. 60 s sequential read, QD32, 1M blocks (libaio)
#   3. temperature sampled every 2 s during the run (hwmon, falls back to smart-log)
#   4. second SMART snapshot, before -> after diff, verdict
# Raw logs are kept in /tmp/nvme_thermal_test.* for later inspection.
set -u

RUNTIME=60
if [ "$#" -ge 1 ] && [ "$1" -eq "$1" ] 2>/dev/null; then
    RUNTIME=$1
    shift
fi

DISKS=("$@")
if [ ${#DISKS[@]} -eq 0 ]; then
    mapfile -t DISKS < <(ls -1 /dev/nvme*n1 2>/dev/null | sort)
fi

if [ "$(id -u)" -ne 0 ]; then
    exec sudo -- "$0" "$RUNTIME" "${DISKS[@]}"
fi

for tool in fio nvme; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: $tool not found"; exit 1; }
done
[ ${#DISKS[@]} -gt 0 ] || { echo "ERROR: no NVMe devices found"; exit 1; }

TMP=$(mktemp -d /tmp/nvme_thermal.XXXXXX)
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null' EXIT

# helpers ------------------------------------------------------------------
# temperature in Celsius for an nvme controller (args: /dev/nvmeX nvmeX).
# hwmon first (millidegrees, no ioctls); smart-log fallback parses both
# "40 C" / "40.5 C" and Kelvin "313 K" formats.
temp_c() {
    local hw v
    hw=$(cat /sys/class/nvme/"$2"/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
    if [ -n "$hw" ]; then
        awk -v m="$hw" 'BEGIN{printf "%.1f", m/1000}'
        return 0
    fi
    v=$(nvme smart-log "$1" 2>/dev/null | grep -m1 -oP '[0-9]+(\.[0-9]+)?(?= C)')
    if [ -n "$v" ]; then
        printf '%s' "$v"
        return 0
    fi
    v=$(nvme smart-log "$1" 2>/dev/null | grep -m1 -oP '[0-9]+(?= K)')
    [ -n "$v" ] && awk -v k="$v" 'BEGIN{printf "%.1f", k-273.15}'
}
sfield() {  # last value of a smart-log field, tolerant to vendor formatting
    sed -n "s/^[^:]*$(printf '%s' "$2" | sed 's/\./\\./g')[[:space:]]*:[[:space:]]*//p" "$1" | head -1
}
num() { tr -cd '0-9.'; }

run_disk() {
    local disk=$1
    local ctrl model nvmename pdev
    ctrl="/dev/$(basename "$disk" | sed 's/n[0-9]*$//')"
    nvmename=$(basename "$ctrl")
    model=$(lsblk -dn -o MODEL "$disk" 2>/dev/null | xargs)
    pdev=$(basename "$(readlink -f "/sys/class/nvme/$nvmename/device" 2>/dev/null)" 2>/dev/null)

    echo
    echo "==================================================================="
    echo "Disk:      $disk ($model)"
    echo "SMART via: $ctrl"
    if [ -n "$pdev" ] && [ -d "/sys/bus/pci/devices/$pdev" ]; then
        echo "PCI:       $pdev  link max=$(cat /sys/bus/pci/devices/$pdev/max_link_speed) x$(cat /sys/bus/pci/devices/$pdev/max_link_width), current=$(cat /sys/bus/pci/devices/$pdev/current_link_speed) x$(cat /sys/bus/pci/devices/$pdev/current_link_width)"
    fi
    echo "Test:      seq read, bs=1M, QD32 (libaio), ${RUNTIME}s"
    echo "==================================================================="

    nvme smart-log "$ctrl" >"$TMP/before.txt" 2>/dev/null
    local t0 cw0
    t0=$(temp_c "$ctrl" "$nvmename")
    cw0=$(sfield "$TMP/before.txt" "critical_warning" | num)

    : >"$TMP/temp.log"
    (
        while :; do
            echo "$(date +%s) $(temp_c "$ctrl" "$nvmename")" >>"$TMP/temp.log"
            sleep 2
        done
    ) &
    SAMPLER=$!

    fio --name=seqread --filename="$disk" --rw=read --bs=1M \
        --iodepth=32 --ioengine=libaio --direct=1 --numjobs=1 \
        --time_based --runtime="$RUNTIME" --group_reporting \
        --log_avg_msec=2000 --write_bw_log="$TMP/bw" >"$TMP/fio.txt" 2>&1
    local rc=$?
    kill "$SAMPLER" 2>/dev/null
    wait "$SAMPLER" 2>/dev/null
    SAMPLER=""

    nvme smart-log "$ctrl" >"$TMP/after.txt" 2>/dev/null
    local t1 cw1
    t1=$(temp_c "$ctrl" "$nvmename")
    cw1=$(sfield "$TMP/after.txt" "critical_warning" | num)

    echo
    echo "--- fio (key lines) ---"
    grep -E 'note:|^  read:|READ: bw=|IO depths|32=100' "$TMP/fio.txt"
    if [ "$rc" -ne 0 ]; then
        echo "WARNING: fio exit code $rc, tail of output:"
        tail -5 "$TMP/fio.txt"
        return 0
    fi

    echo
    echo "--- bandwidth over time (2 s windows) ---"
    local bwlog
    bwlog=$(ls "$TMP"/bw* 2>/dev/null | head -1)
    if [ -n "$bwlog" ] && [ -s "$bwlog" ]; then
        awk -F, '{if (substr($0,1,1)==";") next; if ($2+0>0) printf "  %6.0f s  %7.0f MiB/s\n", $1/1000, $2/1024}' "$bwlog"
    else
        echo "  (bw log not written; raw dir contents below)"
        ls -la "$TMP"
    fi

    echo
    echo "--- temperature during test (every 2 s) ---"
    if [ -s "$TMP/temp.log" ] && [ "$(awk '$2+0>0' "$TMP/temp.log" | wc -l)" -gt 0 ]; then
        local tstart
        tstart=$(head -1 "$TMP/temp.log" | awk '{print $1}')
        awk -v s="$tstart" '$2+0>0 {printf "  %6.0f s  %6.1f C\n", $1-s, $2}' "$TMP/temp.log"
    else
        echo "  (temperature unavailable: no hwmon and smart-log parse failed)"
        nvme smart-log "$ctrl" 2>/dev/null | grep -iE 'temperature|temp' | head -3
    fi

    echo
    echo "--- SMART counters (before -> after) ---"
    local f b a
    for f in critical_warning percentage_used available_spare \
             warning_temp_time critical_comp_time \
             "Thermal Temp.1 Transition Count" "Thermal Temp.1 Total Time" \
             "Thermal Temp.2 Transition Count" "Thermal Temp.2 Total Time" \
             "Thermal Temp.3 Transition Count" "Thermal Temp.3 Total Time"; do
        b=$(sfield "$TMP/before.txt" "$f")
        a=$(sfield "$TMP/after.txt" "$f")
        printf '  %-34s %s -> %s\n' "$f" "${b:-n/a}" "${a:-n/a}"
    done

    # verdict
    # Ramp-up skews min/max: the minimum often sits in the first seconds.
    # Throttling means the SUSTAINED level falls while temp rises, so compare
    # the early sustained window (10-25 s) against the late one (last 15 s).
    local stats=""
    if [ -n "$bwlog" ] && [ -s "$bwlog" ]; then
        stats=$(awk -F, -v rt="$RUNTIME" '
            substr($0,1,1)==";" {next}
            {   t = $1 / 1000; v = $2 / 1024
                if (v <= 0) next
                n++
                if (n == 1 || v < mn) mn = v
                if (v > mx) mx = v
                s += v
                if (t >= 10 && t <= 25) { e += v; en++ }
                if (t >= rt - 15)       { l += v; ln++ }
            }
            END { if (en && ln) printf "%.0f %.0f %.0f %.0f %.0f", mn, mx, s/n, e/en, l/ln }
        ' "$bwlog" 2>/dev/null)
    fi
    echo
    echo "--- VERDICT ---"
    if [ -n "$stats" ]; then
        awk -v st="$stats" -v t0="$t0" -v t1="$t1" -v cw="$cw1" 'BEGIN {
            split(st, s, " "); mn = s[1]; mx = s[2]; avg = s[3]; early = s[4]; late = s[5]
            trend = (early > 0) ? (late - early) / early * 100 : 0
            dt = t1 - t0
            printf "  bw avg %.0f MiB/s (min %.0f, max %.0f); sustained early %.0f, late %.0f MiB/s (trend %+.0f%%); temp %.1f -> %.1f C (%+.1f)\n", avg, mn, mx, early, late, trend, t0, t1, dt
            if (cw + 0 != 0)
                print "  THERMAL THROTTLING ACTIVE NOW (critical_warning != 0)"
            else if (trend <= -25 && dt >= 5)
                print "  THERMAL THROTTLING SUSPECTED: sustained bw fell " (-trend) "% while temp rose +" dt " C"
            else if (trend >= 25)
                print "  RAMP-UP, NOT THROTTLING: bw still climbing; steady-state ~" late " MiB/s"
            else if (trend > -15 && trend < 15)
                print "  BANDWIDTH STABLE - no throttling; ~" late " MiB/s is the real sustained level"
            else
                print "  BW DRIFTS " trend "% WITHOUT throttling flags - retest from idle-cooled state"
        }'
    else
        echo "  (no bw samples for verdict; see fio summary above)"
    fi
    echo
}

echo "NVMe thermal/bandwidth diagnostic, $(date)"
echo "runtime per disk: ${RUNTIME}s"
echo "raw logs: $TMP (kept for inspection)"
for d in "${DISKS[@]}"; do
    run_disk "$d"
done
echo "Done. High 'Thermal Temp.N Total Time' growth + bw drop + temp rise = throttling confirmed."
echo "Remedy: heatsink on the M.2 slot and better case airflow."
