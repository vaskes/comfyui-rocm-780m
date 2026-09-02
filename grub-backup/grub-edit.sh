#!/bin/bash
# Script that did the GRUB edit on 2026-09-02 to fix gfx1103 KFD-queue-eviction MES hang.
# Use as reference — run manually, do not auto-execute.

set -e

# Backup
mkdir -p /opt/mavis-backups/grub-2026-09-02-cwsr-fix
cp /etc/default/grub /opt/mavis-backups/grub-2026-09-02-cwsr-fix/grub.original
cp /boot/grub/grub.cfg /opt/mavis-backups/grub-2026-09-02-cwsr-fix/grub.cfg.before
cat /proc/cmdline > /opt/mavis-backups/grub-2026-09-02-cwsr-fix/cmdline.before.txt

# Add the new parameters to GRUB_CMDLINE_LINUX
# Using sed (manual edit is also fine)
# Append the parameters at the end of the existing GRUB_CMDLINE_LINUX line
NEW_PARAMS="amdgpu.cwsr_enable=0 amdgpu.mes_kiq=1 amdgpu.noretry=1 amdgpu.sg_display=0 amdgpu.gpu_recovery=1 ttm.page_pool_size=6291456 transparent_hugepage=always"

# Backup /etc/default/grub before edit
cp /etc/default/grub /etc/default/grub.bak.$(date +%s)

# Use sed to append to the GRUB_CMDLINE_LINUX="..." line
sed -i "s/^\(GRUB_CMDLINE_LINUX=\".*\) \"/\1 $NEW_PARAMS\"/" /etc/default/grub
sed -i "s/^\(GRUB_CMDLINE_LINUX=\"\)\"/\1$NEW_PARAMS\"/" /etc/default/grub

# Verify the change
echo "=== /etc/default/grub (after edit) ==="
grep GRUB_CMDLINE_LINUX /etc/default/grub
echo
echo "=== Now run: sudo update-grub && sudo reboot ==="
