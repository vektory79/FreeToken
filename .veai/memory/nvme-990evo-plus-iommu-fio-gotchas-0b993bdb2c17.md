---
name: "nvme-990evo-plus-iommu-fio-gotchas"
description: "Samsung 990 EVO Plus NVMe: sustained 6.4 GB/s (7.25 = SLC burst); fio libaio, dd caps 1.5; IOMMU tax = old 4 GB/s"
type: project
lastUpdated: 2026-09-12T22:11
lastRecall: 2026-09-12T22:05
---

# NVMe on work.vektory79.me rig: 990 EVO Plus final verdict + benchmark methodology gotchas

## Disks (as of 2026-09-12)
- nvme1n1: Samsung 990 EVO Plus 2TB (PM9C1a), spec 7250 MB/s seq read. On PCH root port 00:1d.0, link Gen4 x4 (~15 GB/s raw, 2x headroom). NO ATS (nothing on `lspci -vv` grep) -> every DMA translated at the root complex under this rig's IOMMU (see rtx5090-pcie-gen5-bw-cap).
- nvme0n1: Samsung 980 1TB (DRAM-less Gen3), spec ~3.5 GB/s.
- Root FS is on SATA (sda5), so raw-device fio reads on the NVMes are safe.

## History and root-cause story
User previously had the 990 EVO Plus in a slot near the GPU and saw ~4 GB/s vs the 7.25 spec; he moved it suspecting a GPU conflict. That conflict theory is dead: the GPU caps even alone, translation is per-device. After the iommu=pt discovery (rtx5090-pcie-gen5-bw-cap), the leading explanation for the old 4 GB/s is the SAME Translated-IOMMU ~50% tax (proven on the GPU: 46 -> 25 GB/s) applied to a disk that really sustains ~6-6.4 GB/s -> ~3-4 GB/s. Still NOT retrospectively verified; the old numbers were dd-based (unreliable, below). Retro-verify only by moving the disk back to the old slot + rerunning the same fio test.

## FINAL verdict (2026-09-12, 60s libaio runs, diagnostic script .tasks/pcie-bw-probe/nvme_thermal_test.sh)
- 990 EVO Plus: NO thermal throttling. Bandwidth RISES over the run: ~12s ramp-up (DRAM-less HMB warm-up, 1.8-3.4 GB/s) then sustained climbing 5.9 -> 6.36 GB/s by 60s. Temp 47 -> 73 C is plain load heating, far below Samsung's throttle zone (~77-84 C); critical_warning 0, available_spare 100%, percentage_used 0%. The 7.25 GB/s spec is SLC-cached short burst; real TLC sustained ~6-6.4.
- Samsung 980: same ramp shape, 1.06 -> 2.4 GB/s. Sustained ~2-2.4 GB/s = normal DRAM-less Gen3, not a platform fault (990 EVO Plus on the same PCH did 5.5+).
- 20s runs read low when the drive is warm from a prior test (4.07 GB/s once, opposite of the thermal-slide prediction); use >=60s and compare early vs late sustained windows.
- Samsung exposes no vendor thermal-time SMART fields (n/a).

## Benchmark methodology gotchas (durable)
- fio WITHOUT --ioengine=libaio silently uses psync and caps queue depth at 1 even with --iodepth=32 (output warns "synchronous I/O engine ... capped at 1"; "IO depths: 1=100%"). Measured QD1 on the 990 EVO Plus: 1.49 GB/s at ~700 us - normal QD1, NOT the drive spec.
- dd is synchronous too: ~1.5-1.8 GB/s physical ceiling on this disk; historic dd numbers can never show the 7.25 GB/s spec. Never verify NVMe seq bandwidth with dd.
- Correct command: sudo fio --name=seqread --filename=/dev/nvme1n1 --rw=read --bs=1M --iodepth=32 --ioengine=libaio --direct=1 --numjobs=1 --time_based --runtime=60 --group_reporting. Expect sustained ~6-6.4 GB/s after the ramp-up, not 7.25.

## Diagnostic-script lessons (.tasks/pcie-bw-probe/nvme_thermal_test.sh, v2 bugs fixed)
- Temp sampling: use hwmon and parse both smart-log C and K formats (old regex missed Samsung's "N C" integer format -> all temps 0.0).
- bw-log glob must be bw* (a *.log glob misses fio's per-window files).
- Report the DEVICE bdf via basename, not dirname (dirname showed the PCH root port instead of the device).
- Verdict logic must compare early (10-25s) vs late (last 15s) sustained windows; (max-min)/max false-positives on the ramp-up minimum.

## Slot caution (if ever moving disks again)
Check the MSI PRO Z790-P manual whether that M.2 shares lanes with PCI_E1 (GPU would drop to x8); verify via /sys/bus/pci/devices/0000:01:00.0/current_link_width.
