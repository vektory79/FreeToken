---
name: "rtx5090-pcie-gen5-bw-cap"
description: "RTX 5090 Gen5 DMA cap root cause: IOMMU Translated; iommu=pt restores 46/57 GB/s; rig details and probe artifacts"
type: project
lastUpdated: 2026-09-13T23:42
lastRecall: 2026-09-15T19:57
---

# RTX 5090 PCIe Gen5 bandwidth cap on work.vektory79.me - ROOT CAUSE: IOMMU Translated mode (fixed 2026-09-12)

## Rig
- Palit RTX 5090 (10de:2b85, sub f318) at 01:00.0, direct in CPU PEG port 00:01.0 (no riser)
- Board: MSI PRO Z790-P WIFI (MS-7E06), BIOS A.H0 2025-06-04
- i7-14700KF, 4x48 GiB DDR5-5600, kernel 7.0.0-31-generic, driver 610.43.02, CUDA 13.3, nvbandwidth v0.10.0 in /media/ai/src/nvbandwidth/build

## Root cause (confirmed by A/B reboot test)
Linux kernel IOMMU default domain = Translated (DMAR present, GPU has NO ATS -> every DMA translated at root complex). This HALVED CE DMA bandwidth on Gen5 x16.
- Translated: H2D 25.3, D2H 23.6 GB/s, bidirectional sum ~29.5 (vs ~49 if limits were per-direction), SM path 7-11 GB/s, chase latency 617 ns
- iommu=pt: H2D 46.2 (+82%), D2H 57.1 GB/s (+142%, = Gen5 x16 spec), chase latency 601 ns (unchanged)

## Final state
- GRUB has only "iommu=pt" (both intel_iommu=off and iommu=pt also works identically); 18 iommu groups = IOMMU alive; H2D 46.20 / D2H 57.12 GB/s. Case closed 2026-09-12.
- MRRS setpci 256->4096 flat both under Translated and passthrough -> left at default 256. GPU DevCap MaxPayload 256 is the card's hardware max.
- H2D ~46 (~73% wire) vs D2H ~57 (~90%) asymmetry = non-posted read round-trips vs posted writes on Intel client PEG; platform characteristic, not actionable.
- Falsified: link degradation, AER, DRAM BW, buffer-size effect 4MiB-4GiB, riser, BIOS age.
- Non-factors (explained, details in article): VF BAR "can't assign; no space" = SR-IOV window issue (GPU has 1 VF, irrelevant to DMA); ASPM off via FADT + root port LnkCap lacks ASPM (power feature only).

## Durable lesson
On consumer Intel + IOMMU Translated default, any ATS-less DMA device can be capped at Gen4-like levels while the link trains Gen5 perfectly; pointer-chase latency may NOT change when fixed (RTT-dominated). Applies to NVMe too - same-rig case fully covered in nvme-990evo-plus-iommu-fio-gotchas (990 EVO Plus sustained 6.4 GB/s; old ~4 GB/s most plausibly the same Translated tax).

## Artifacts
- Full article (fix steps, verification checklist, false hypotheses, measurements): /media/ai/src/FreeToken/.veai/docs/rtx5090-pcie-gen5-iommu-bandwidth.md
- Probe scripts: /media/ai/src/FreeToken/.tasks/pcie-bw-probe/ (pcie_bw_probe.py, size_scaling.py, dram_dir_bw.py, pcie_bw.cu)
