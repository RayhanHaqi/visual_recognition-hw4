#!/usr/bin/env bash
# Recover RTX 5090 (PCI 21:00.0) after CUDA crash / "Unknown Error" in nvidia-smi.
# Run on w61:  bash scripts/recover_gpu_5090.sh
# Soft recovery needs sudo; full power-off may still be required (Xid 79).
set -euo pipefail

BROKEN_PCI="${BROKEN_PCI:-0000:21:00.0}"
OK_PCI="${OK_PCI:-0000:e1:00.0}"

echo "=== 1) PCI devices (expect 2 NVIDIA VGA entries) ==="
lspci -nn | grep -i nvidia || true
echo

echo "=== 2) nvidia-smi (before recovery) ==="
nvidia-smi -L 2>&1 || true
echo

echo "=== 3) Recent NVIDIA kernel errors ==="
if command -v dmesg >/dev/null 2>&1; then
  sudo dmesg -T 2>/dev/null | grep -iE 'NVRM|Xid|nvidia|21:00' | tail -30 || true
fi
echo

echo "=== 4) Processes using NVIDIA devices ==="
if command -v fuser >/dev/null 2>&1; then
  sudo fuser -v /dev/nvidia* 2>/dev/null || echo "(no processes on /dev/nvidia*)"
else
  echo "fuser not installed; skip"
fi
echo

echo "=== 5) Stop common GPU users (ignore errors) ==="
pkill -f 'train_hw4.py' 2>/dev/null || true
pkill -f 'python.*inference.py' 2>/dev/null || true
sleep 2
echo

echo "=== 6) GPU reset (needs sudo) ==="
# Broken card is usually GPU index 0 when 4090 is GPU 1.
for idx in 0 1; do
  echo "Trying nvidia-smi --gpu-reset -i ${idx} ..."
  if sudo nvidia-smi --gpu-reset -i "${idx}" 2>/dev/null; then
    echo "  reset OK for GPU ${idx}"
  else
    echo "  reset failed or skipped for GPU ${idx}"
  fi
done
echo

echo "=== 7) Reload driver modules (needs sudo; fails if GPUs still busy) ==="
if sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null; then
  sudo modprobe nvidia
  echo "  driver reloaded"
else
  echo "  could not unload nvidia modules (reboot or power-off required)"
fi
echo

echo "=== 8) nvidia-smi (after recovery) ==="
if nvidia-smi -L; then
  echo
  echo "SUCCESS: all listed GPUs respond."
  nvidia-smi
  exit 0
fi

echo
echo "FAILED: 5090 still broken."
echo "Next steps (in order):"
echo "  1) sudo reboot"
echo "  2) If still broken: FULL POWER OFF (hold power 10s), wait 30s, power on"
echo "  3) Check: sudo dmesg | grep -i xid"
echo "  4) If Xid 79 repeats under load: PSU/power cables, thermals, BIOS PCIe Gen3, avoid dual-GPU training"
echo "Homework: use only 4090 (export CUDA_VISIBLE_DEVICES=1) until after deadline."
exit 1
