# TTM / GPU memory parameters — что есть что и как считать

Эта шпаргалка для тех, кто меняет GTT/VRAM на AMD APU (Radeon 780M и подобных) и не понимает, почему GRUB ругается или ComfyUI не видит память.

## TL;DR

Три параметра в GRUB должны быть согласованы друг с другом и с реальным размером GTT. Если не согласованы — kernel откажется выделить память или torch увидит обрезанный GTT.

```bash
# Формула для gfx1103 (APU, 4 KB страницы):
pages = GTT_MiB × 1024 × 1024 / 4 / 1024  # = GTT_MiB × 256
```

## Параметры

### 1. `amdgpu.gttsize=NNNNN`

**Что это:** размер GTT в MiB. GTT (Graphics Translation Table) — основная память, в которую amdgpu пишет тензоры на APU (потому что VRAM на iGPU = системная RAM, выделенная под GPU, а реальная HBM нет).

**Кто использует:** amdgpu driver при инициализации GPU. Это HARD CAP на размер GTT.

**Как выбрать:**
- По умолчанию = 1/2 system RAM (т.е. ~30 GB на 64 GB RAM)
- Для нашего сетапа 51 GB — 4 GB VRAM + 51 GB GTT = 55 GB GPU memory, остальное ~3 GB system
- Не больше system RAM минус то что хочешь оставить системе
- Менять: GRUB → update-grub → reboot

**Проверить после ребута:**
```bash
echo $(( $(cat /sys/class/drm/card1/device/mem_info_gtt_total) / 1024 / 1024 ))  # MiB
```

### 2. `ttm.pages_limit=NNNNN`

**Что это:** hard limit на количество TTM pages, которые kernel может выделить глобально. Page = 4 KB.

**Кто использует:** TTM (Translation Table Maps) subsystem в kernel. Это SOFT HARD CAP — TTM не даст выделить больше страниц чем pages_limit, даже если физической памяти хватает.

**Правило:** `ttm.pages_limit × 4 KB ≥ amdgpu.gttsize`

**Почему именно GTT:** TTM обслуживает VRAM + GTT + system RAM pages. Если pages_limit < GTT, kernel не сможет полностью использовать GTT (тензоры будут медленно фоллбэчиться в swap или вообще OOM).

**Как выбрать:**
```bash
# Формула (4 KB pages):
pages_limit = ceil(gttsize_MiB × 1024 × 1024 / 4096)
            = gttsize_MiB × 256
```

| GTT (MiB) | pages_limit (pages) |
|---|---|
| 40,960 (40 GB) | 10,485,760 |
| 51,200 (50 GB) | 13,107,200 |
| 52,224 (51 GB) | 13,369,344 |
| 92,160 (90 GB) | 23,592,960 |
| 108,544 (106 GB) | 27,787,264 |

**Проверить после ребута:**
```bash
cat /sys/module/ttm/parameters/pages_limit  # в страницах
echo $(( $(cat /sys/module/ttm/parameters/pages_limit) * 4 / 1024 ))  # в MiB
```

### 3. `ttm.page_pool_size=NNNNN`

**Что это:** начальный размер пула свободных страниц для быстрого аллоцирования. Тоже в страницах (4 KB). Может расти до pages_limit по необходимости.

**Кто использует:** TTM subsystem. Это PERFORMANCE HINT, не hard cap. Больше пул = меньше TTM тратит на eviction при нагрузке.

**Правило:** `ttm.page_pool_size ≤ ttm.pages_limit`

**Как выбрать:**
- Default Linux = что-то около 1/8 system RAM в страницах
- Для GPU-нагрузки (большие тензоры): ~50-60% от pages_limit
- Слишком большой pool = зря держит RAM заблокированной
- Слишком маленький = TTM будет часто вытеснять страницы, тормоза

**Формула для APU с большим GTT:**
```bash
page_pool_size ≈ pages_limit × 0.6
```

| GTT (MiB) | pages_limit | page_pool_size (60%) | в MiB |
|---|---|---|---|
| 40,960 (40 GB) | 10,485,760 | 6,291,456 | 24,576 (24 GB) |
| 52,224 (51 GB) | 13,369,344 | 8,021,606 | 31,334 (30.6 GB) |

**Проверить после ребута:**
```bash
cat /sys/module/ttm/parameters/page_pool_size  # в страницах
echo $(( $(cat /sys/module/ttm/parameters/page_pool_size) * 4 / 1024 ))  # в MiB
```

## Полный рабочий пример (наш 780M, 64 GB RAM)

```bash
# GRUB (/etc/default/grub, GRUB_CMDLINE_LINUX_DEFAULT)
"quiet splash amd_iommu=on iommu=pt ttm.pages_limit=13369344 ttm.page_pool_size=8021606 amdgpu.gttsize=52224 amdgpu.lockup_timeout=60000 amdgpu.cwsr_enable=0 amdgpu.mes_kiq=1 amdgpu.noretry=1 amdgpu.sg_display=0 amdgpu.gpu_recovery=1 transparent_hugepage=always"
```

| Параметр | Значение | Что значит |
|---|---|---|
| `ttm.pages_limit=13369344` | 51 GiB в страницах | TTM hard cap = 51 GB |
| `ttm.page_pool_size=8021606` | ~30.6 GiB в страницах | initial pool = 60% от limit |
| `amdgpu.gttsize=52224` | 51 GiB в MiB | driver GTT = 51 GB |
| `amdgpu.cwsr_enable=0` | 0 | CWSR fix для gfx1103 (ROCm issues #5590, #5665) |
| `amdgpu.mes_kiq=1` | 1 | MES KIQ on (workaround для KFD queue eviction) |
| `amdgpu.noretry=1` | 1 | не ретраить при GPU error |
| `amdgpu.gpu_recovery=1` | 1 | включить GPU recovery (но не использовать — hang требует cwsr fix) |
| `amdgpu.lockup_timeout=60000` | 60000 ms | timeout для lockup detection (60s) |
| `amdgpu.sg_display=0` | 0 | отключить scatter-gather display (помогает stability) |
| `amd_iommu=on` | on | IOMMU on (НЕ off!) |
| `iommu=pt` | passthrough | pass-through mode (быстрее) |
| `transparent_hugepage=always` | always | THP on (производительность) |

## Как менять GTT

**Шаг 1.** Решить новый размер (например, с 40 GB на 51 GB).

**Шаг 2.** Пересчитать pages_limit и page_pool_size:

```bash
NEW_GTT_MiB=52224

# pages_limit
NEW_PAGES=$(( NEW_GTT_MiB * 256 ))
echo "ttm.pages_limit=$NEW_PAGES"

# page_pool_size (60% от pages_limit)
NEW_POOL=$(( NEW_PAGES * 60 / 100 ))
echo "ttm.page_pool_size=$NEW_POOL"
```

**Шаг 3.** Обновить GRUB:

```bash
sudo cp /etc/default/grub /etc/default/grub.bak
sudo sed -i "s/amdgpu.gttsize=[0-9]*/amdgpu.gttsize=$NEW_GTT_MiB/" /etc/default/grub
sudo sed -i "s/ttm.pages_limit=[0-9]*/ttm.pages_limit=$NEW_PAGES/" /etc/default/grub
sudo sed -i "s/ttm.page_pool_size=[0-9]*/ttm.page_pool_size=$NEW_POOL/" /etc/default/grub
sudo update-grub
```

**Шаг 4.** Reboot и проверить:

```bash
sudo reboot
# после ребута:
echo "GTT (MiB):  $(($(cat /sys/class/drm/card1/device/mem_info_gtt_total)/1024/1024))"
echo "VRAM (MiB): $(($(cat /sys/class/drm/card1/device/mem_info_vram_total)/1024/1024))"
echo "pages_limit (MiB): $(($(cat /sys/module/ttm/parameters/pages_limit)*4/1024))"
echo "page_pool_size (MiB): $(($(cat /sys/module/ttm/parameters/page_pool_size)*4/1024))"
```

## Симптомы неправильных значений

| Симптом | Причина | Фикс |
|---|---|---|
| `Total VRAM 40 GB` вместо ожидаемых 51 GB | `ttm.pages_limit` < `gttsize` | Увеличить `ttm.pages_limit` |
| ComfyUI падает с OOM на модели 30 GB при GTT 51 GB | `ttm.page_pool_size` слишком мал | Увеличить pool (TTM постоянно evictит) |
| dmesg: `ttm: Maximum number of pages exceeded` | pages_limit слишком мал | Увеличить pages_limit |
| dmesg: `amdgpu: Not enough memory for GTT` | gttsize > доступной RAM | Уменьшить gttsize |
| GTT total показывает меньше чем в GRUB | driver применил не все параметры | проверить через sysfs, ребут |
| После `amdgpu.gttsize=N` GTT остался старый | не ребутнулся, или GRUB не обновлён | update-grub, reboot |

## Шпаргалка для 4 KB pages

```text
1 MiB = 256 pages
1 GiB = 262144 pages
1 GB (decimal) ≈ 238419 pages
pages = MiB × 256
MiB = pages ÷ 256
```

## Полезные команды

```bash
# Текущие значения (всё в MiB, читаемо):
echo "VRAM: $(($(cat /sys/class/drm/card1/device/mem_info_vram_total)/1024/1024)) MiB"
echo "GTT:  $(($(cat /sys/class/drm/card1/device/mem_info_gtt_total)/1024/1024)) MiB"
echo "pages_limit:  $(($(cat /sys/module/ttm/parameters/pages_limit)*4/1024)) MiB"
echo "page_pool:    $(($(cat /sys/module/ttm/parameters/page_pool_size)*4/1024)) MiB"

# Размер страницы:
getconf PAGESIZE  # обычно 4096

# Что показывает ComfyUI / torch:
docker exec comfyiu python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
print(f'torch sees: {p.total_memory/1024**3:.2f} GB')
"
```
