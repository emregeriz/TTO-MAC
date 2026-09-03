# TTO-MAC — Trento Toplu Okuma (macOS)

Palet üstündeki kasaları 6 görüntüden sayan uygulamanın **macOS sürümü**.
Windows'ta 6 endüstriyel kamerayla çalışan sistemin, Mac'te **hız ölçümü ve
arayüz denemesi** için hazırlanmış portu.

> **Claude'a not:** Bu dosyayı okuyup kurulumu yapman isteniyorsa doğrudan
> [Hızlı kurulum](#hızlı-kurulum) bölümündeki komutu çalıştır. Takılırsan
> [Sorun giderme](#sorun-giderme) bölümüne bak. Kullanıcının Mac'inde Python
> bile kurulu olmayabilir — `kur.sh` bunu da halleder.

---

## Hızlı kurulum

Terminal'i aç, bu klasöre gir ve tek komutu çalıştır:

```bash
chmod +x kur.sh baslat.sh
./kur.sh
```

Betik sırayla şunları yapar (kurulu olanları atlar):

1. **Xcode Command Line Tools** — derleyiciler
2. **Homebrew** — paket yöneticisi
3. **Python 3.11 + `python-tk@3.11`** — ⚠️ Tk ayrı pakettir, o olmadan arayüz açılmaz
4. **`.venv`** sanal ortamı
5. **Zorunlu paketler** (`requirements.txt`) — arayüz, OpenCV, YOLO, zxing barkod
6. **OCR** (`requirements-ocr.txt`) — opsiyonel; kurulamazsa uygulama yine çalışır
7. **Doğrulama** — her paketi tek tek import edip raporlar

Kurulum bitince:

```bash
./baslat.sh
```

---

## Test etmek

Depoda **`ornek_goruntuler/`** klasöründe 6 gerçek kamera görüntüsü var
(`kamera1.jpg` … `kamera6.jpg`). Kamera bağlamadan bunlarla test edilir.

1. `./baslat.sh` ile uygulamayı aç.
2. Açılışta **Sevkiyat** ekranı gelir → İrsaliye no ve kasa adedi (örn. `8074`)
   girip **SÜRECİ BAŞLAT**. (Süreç açmadan denemek için
   *"Sevkiyat açmadan kameraları kontrol et →"*.)
3. **YENİ PALET — 6 GÖRSEL SEÇ** düğmesine bas.
4. Açılan pencere `ornek_goruntuler/` klasöründe açılır → **6 görseli birden
   seç** (⌘A) → Aç.
5. Görüntüler sırayla işlenir; her biri için log'a süre yazılır, sonunda
   **⏱ TOPLAM SÜRE** satırı çıkar. Sayım bitince doğrulama ekranı açılır.

**Tek tek yüklemek istersen:** her kamera kartındaki **Yükle** düğmesi tek bir
görüntüyü o kameraya atar.

### Beklenen sonuç

Bu 6 görüntüde referans değerler (Windows'ta ölçülen):

| | değer |
|---|---|
| Ham tespit (6 kamera toplamı) | ~312 kutu |
| Çakışan (tekilleştirilen) | ~56 |
| **Gerçek kasa sayısı** | **~256** |
| Okunan barkod | ~290 |

Sayı birkaç kasa oynayabilir — barkod motoru Windows'takinden farklı
(aşağıya bakın).

---

## macOS'ta neler farklı

| | Windows | macOS |
|---|---|---|
| **Kamera** | 6 Hikrobot, MVS SDK | ❌ Yok — MVS SDK'nın macOS sürümü yok. Sayım **görüntü dosyalarından** yapılır |
| **Barkod** | Aremak Code Reader (ücretli .NET DLL + USB dongle) | **zxing-cpp** (açık kaynak). Dongle ve .NET macOS'ta çalışmaz |
| **YOLO** | NVIDIA CUDA | Apple Silicon'da **MPS** (Metal), Intel Mac'te CPU |
| **OCR** | PaddleOCR, CUDA | PaddleOCR, **CPU** (Paddle'ın Mac'te GPU desteği yok) |

**Barkod farkı önemli:** zxing, Aremak'a göre biraz daha az barkod okur.
Okunamayan kasalar kırmızı kalır ve OCR/elle işaretlemeye düşer — sayım yine
doğru çalışır, sadece daha fazla kasa "okunamayan" olur. Bu port **hız
ölçümü** içindir; sahada Aremak'lı Windows sürümü kullanılır.

---

## Uygulama ne yapıyor

**Sevkiyat süreci:** Bir tır = bir süreç. İrsaliye no/plaka/adet ile açılır,
tırdaki paletler tek tek okunup onaylandıkça sürece eklenir (272 → 522 → …),
operatör bitirene kadar açık kalır.

**Sayım zinciri:** YOLO ile kasa kutuları bulunur → her kasada barkod okunur →
arka plan/hatalı kutular elenir → 6 kamera arası tekilleştirme yapılır →
okunamayan kasalarda OCR ile numaradan seri tespit edilir → doğrulama ekranı.

**Tekilleştirme** ortak barkodlardan kamera çiftlerinin kaymasını öğrenir;
her kasa, gören kameralardan en küçük numaralısına yazılır (kamera paylarının
toplamı = tekil toplam).

**Filtreler:** genişlik (%85 eşiği), yükseklik (komşularının eğilimine göre
−%20/+%35; kare kenarında kesilenlerde −%8), sütun dışı kutular.

---

## Klasör yapısı

```
TTO-MAC/
├── kur.sh                  kurulum (tek komut)
├── baslat.sh               başlatıcı
├── requirements.txt        zorunlu paketler
├── requirements-ocr.txt    opsiyonel OCR paketleri
├── app.py                  arayüz + sevkiyat süreci
├── detection.py            YOLO tespiti + zxing barkod
├── aggregator.py           kameralar arası tekilleştirme
├── mac_camera.py           kamera yöneticisi vekili (macOS'ta cihaz yok)
├── ocr_reader.py           PaddleOCR ile seri tespiti
├── ocr_engine.py, app_ui.py   PaddleOCR sarmalayıcıları
├── widgets.py, theme.py, logo.py
├── models/V8LAST.pt        YOLO modeli (22 MB)
├── ornek_goruntuler/       6 test görüntüsü
└── captures/               çıktılar (git'e girmez)
```

---

## Sorun giderme

**`No module named _tkinter`**
Tk kurulu değil. Homebrew Python'u tkinter'i ayrı paketle veriyor:
```bash
brew install python-tk@3.11
rm -rf .venv && ./kur.sh
```

**Uygulama açılıyor ama pencere boş / yazılar bozuk**
`SF Pro Text` yazı tipi yoksa Tk otomatik düşer; sorun sürerse `theme.py`
içinde `FONT = "Helvetica Neue"` yapın.

**`paddleocr` kurulamadı**
Sorun değil — sayım ve barkod çalışır, yalnız okunamayan kasalarda otomatik
seri tespiti kapalı olur. Elle denemek için:
```bash
.venv/bin/pip install paddlepaddle paddleocr
```

**Barkod okunmuyor / "BARKOD SDK HATASI" şeridi**
zxing kurulu değil demektir:
```bash
.venv/bin/pip install zxing-cpp
```
Uygulamada kırmızı şeritteki **TEKRAR DENE** düğmesine basın.

**YOLO çok yavaş**
Apple Silicon'da MPS kullanılır; log'da *"Apple GPU (MPS) kullanılıyor"*
yazmalı. *"CPU kullanılıyor"* yazıyorsa torch MPS'siz kurulmuştur:
```bash
.venv/bin/pip install --upgrade --force-reinstall torch torchvision
```

**"Kameraları Tara" hiçbir şey bulmuyor**
Beklenen davranış — macOS'ta canlı kamera desteği yok. Araç çubuğundaki
**🖼 Görüntü Klasörü** düğmesi örnek görüntüleri Finder'da açar.

**Model bulunamadı**
`models/V8LAST.pt` dosyasının indiğinden emin olun (22 MB). Git LFS
kullanılmıyor, normal `git clone` ile gelir.

---

## Gereksinimler

- macOS 12+ (Apple Silicon ya da Intel)
- ~4 GB disk (torch + paddle büyük)
- İnternet (ilk kurulum ve OCR modelinin ilk indirilmesi için)
