# LoRA'yı Çok Basit Örneklerle Anlamak

Bu dosya, `lora/` klasöründeki kodu temel alarak LoRA'nın nasıl çalıştığını
sade Türkçe ile anlatır. Amaç formül ezberletmek değil; "modelin tamamını
yeniden eğitmeden küçük bir parça ile davranışı nasıl değiştiriyoruz?" sorusunu
netleştirmektir.

En kısa fikir:

> Büyük modeli donduruyoruz. Yanına küçük bir ayar parçası takıyoruz. Eğitimde
> sadece bu küçük parçayı değiştiriyoruz.

Bu küçük parçaya genelde **adapter** denir.

---

## 1. Önce Sembolleri İnsan Diline Çevirelim

LoRA anlatılırken sık sık `x`, `W`, `A`, `B` gibi semboller görülür. Bunlar
başta soğuk duruyor, ama aslında basit şeyleri temsil ediyor:

| Sembol | Basit anlamı                                                                                     |
| ------ | ------------------------------------------------------------------------------------------------ |
| `x`    | Katmana gelen sayı listesi. Bir harfin, kelimenin veya ara temsilin model içindeki sayısal hali. |
| `W`    | Katmanın büyük ağırlık tablosu. Modelin önceden öğrendiği ana davranış burada durur.             |
| `A`    | LoRA'nın ilk küçük tablosu. Gelen bilgiyi daha küçük bir ara alana indirir.                      |
| `B`    | LoRA'nın ikinci küçük tablosu. Küçük ara bilgiyi tekrar katmanın çıktı boyutuna çıkarır.         |
| `s`    | Adapter etkisinin ses düğmesi gibi düşünülebilecek ölçek katsayısı.                              |

Bu dokümanda ana cümle şu olacak:

```text
son cevap = eski modelin cevabı + LoRA'nın küçük düzeltmesi
```

Teknik kaynaklarda bunun sembollerle yazılmış halini görebilirsin. Bu dosyada
onu şöyle okuyacağız:

```text
eski cevap + ölçeklenmiş küçük adapter düzeltmesi
```

Yani formülün özü, karmaşık görünen semboller değil, şu fikirdir:

> Model eskisi gibi çalışmaya devam eder; LoRA sadece yanına küçük bir düzeltme
> ekler.

---

## 2. LoRA Neyi Çözmeye Çalışıyor?

Bir transformer modelinde birçok katman büyük ağırlık tabloları kullanır. Bu
tabloları, "gelen bilgiyi başka bir sayı listesine dönüştüren büyük karar
tabloları" gibi düşünebilirsin.

Örneğin bir katmanda `4096 x 4096` boyutunda bir ağırlık tablosu varsa, içinde:

```text
4096 * 4096 = 16,777,216
```

adet sayı vardır.

Normal fine-tuning yaparsak bu 16 milyondan fazla sayının hepsini değiştirmeye
izin veririz. Büyük modellerde bu çok pahalıdır:

- Daha fazla bellek gerekir.
- Daha fazla eğitim süresi gerekir.
- Her yeni görev için büyük bir model kopyası saklamak gerekir.

LoRA'nın sorduğu soru şudur:

> Yeni görev için gerçekten bütün büyük tabloyu değiştirmemiz gerekiyor mu?

LoRA'nın cevabı:

> Hayır. Büyük tabloyu sabit bırakıp yanına küçük bir düzeltme eklemek çoğu
> zaman yeterli olur.

---

## 3. LoRA'nın Ana Fikri

Normal bir katmanı şöyle düşün:

```text
girdi -> büyük ağırlık tablosu -> çıktı
```

LoRA eklenince akış iki yola ayrılır:

```text
                     -> büyük ağırlık tablosu -> eski cevap
girdi ---------------|
                     -> küçük LoRA adapteri  -> küçük düzeltme

son cevap = eski cevap + küçük düzeltme
```

Burada kritik nokta:

- Büyük ağırlık tablosu dondurulur.
- Küçük LoRA adapteri eğitilir.
- Eğitim sonunda model yeni davranışı bu küçük parça sayesinde kazanır.

Bu yüzden LoRA, "modeli yeniden eğitmek" değil, daha çok "modele küçük bir ayar
parçası takmak" gibidir.

---

## 4. Çok Basit Sayısal Örnek

Bir katmanın 3 sayı aldığını düşünelim.

Girdi:

```text
[1, 1, 1]
```

Ana modelin eski davranışı çok basit olsun:

```text
Her sayıyı 2 ile çarp.
```

O zaman eski cevap:

```text
[2, 2, 2]
```

Şimdi LoRA adapteri şunu öğrensin:

```text
İlk iki girdiye bak.
Onları topla.
Bu toplamı sadece ilk çıktıya ekle.
```

İlk iki girdinin toplamı:

```text
1 + 1 = 2
```

Adapter etkisini biraz güçlendirelim; ölçek değeri `2` olsun:

```text
2 * 2 = 4
```

LoRA'nın küçük düzeltmesi:

```text
[4, 0, 0]
```

Son cevap:

```text
eski cevap      = [2, 2, 2]
LoRA düzeltmesi = [4, 0, 0]
son cevap       = [6, 2, 2]
```

Burada büyük fikir çok net:

- Ana modelin eski cevabı `[2, 2, 2]` idi.
- LoRA, sadece küçük bir düzeltme ekledi.
- Sonuç `[6, 2, 2]` oldu.
- Ana modelin büyük ağırlıkları değişmedi.

### Bu Örneğin Teknik Karşılığı

Yukarıdaki örnekte LoRA'nın iki küçük tablosu şöyle düşünülebilir:

```text
A: "ilk iki girdiyi oku ve topla"
B: "bu sonucu sadece ilk çıktıya gönder"
```

Yani `A` sıkıştırır, `B` tekrar çıktı tarafına yayar.

Bu yüzden LoRA'daki iki küçük tabloyu şöyle akılda tutabilirsin:

```text
A = daraltan parça
B = genişleten parça
```

---

## 5. Peki Neden İki Küçük Tablo Kullanılıyor?

LoRA doğrudan büyük bir düzeltme tablosu eğitmez. Bunun yerine iki küçük tablo
eğitir:

```text
girdi -> küçük ara alan -> çıktı düzeltmesi
```

Bunu bir huni gibi düşünebilirsin:

```text
büyük bilgi -> dar huni -> küçük ara bilgi -> tekrar genişlet -> düzeltme
```

Buradaki "dar huni" kısmının genişliğine **rank** denir.

Rank küçükse:

- Adapter daha az parametre taşır.
- Eğitim daha ucuz olur.
- Ama adapterin ifade gücü daha sınırlı olur.

Rank büyükse:

- Adapter daha güçlü olur.
- Fakat daha fazla parametre gerekir.

Bu repoda LoRA için genelde rank `4` kullanılıyor. VeRA için rank `64`
kullanılıyor, çünkü VeRA'nın eğittiği parametre sayısı rank artsa bile çok daha
yavaş büyüyor.

---

## 6. Parametre Kazancı Neden Büyük?

Tam bir ağırlık tablosunu eğitmek pahalıdır.

Örnek:

```text
Ana tablo boyutu: 32 x 32
Tam eğitilecek sayı: 32 * 32 = 1024
```

LoRA rank `4` kullanırsa iki küçük tablo eğitir:

```text
İlk küçük tablo:  4 x 32 = 128 sayı
İkinci küçük tablo: 32 x 4 = 128 sayı
Toplam: 256 sayı
```

Yani:

```text
Tam eğitim: 1024 sayı
LoRA:        256 sayı
```

Daha büyük modellerde fark çok daha çarpıcıdır:

```text
Ana tablo boyutu: 4096 x 4096
Tam eğitim:       16,777,216 sayı
LoRA rank 8:      65,536 sayı
```

LoRA'nın pratik değeri buradan gelir: çok daha az sayı eğiterek modelin
davranışını değiştirebilir.

---

## 7. Neden Bir Taraf Sıfır, Bir Taraf Rastgele Başlıyor?

LoRA'da başlangıçta genellikle şu yapılır:

```text
A: küçük rastgele değerlerle başlar
B: tamamen sıfır başlar
```

Bunun sebebi iki isteği aynı anda karşılamaktır.

### 7.1. Başlangıçta Model Bozulmasın

`B` sıfır olduğu için LoRA'nın düzeltmesi de sıfırdır.

Yani eğitim başlamadan önce:

```text
son cevap = eski modelin cevabı
```

Model ilk adımda pretrained halini korur. Bu iyi bir şeydir; çünkü rastgele
bozulmuş bir modelden değil, zaten öğrenmiş bir modelden eğitime başlarız.

### 7.2. Ama Adapter Hareket Edebilsin

Eğer hem `A` hem `B` sıfır başlarsa adapter takılır kalır. Çünkü iki taraf da
birbirini harekete geçirecek sinyali üretemez.

Basit sezgi:

```text
A rastgele, B sıfır -> model başlangıçta bozulmaz, B eğitimle hareket edebilir
A sıfır,   B sıfır -> adapter başlayamaz
```

Bu yüzden LoRA'da "bir taraf rastgele, bir taraf sıfır" başlatılır.

---

## 8. `alpha` ve Ölçek Ne İşe Yarıyor?

LoRA'nın ürettiği küçük düzeltme bazen çok zayıf, bazen çok güçlü olabilir.
`alpha` bu düzeltmenin gücünü ayarlayan bir düğme gibi düşünülebilir.

Klasik LoRA'da adapter etkisi kabaca şöyle ölçeklenir:

```text
ölçek = alpha / rank
```

Örneğin `alpha = 8` ise:

| rank | klasik LoRA ölçeği |
| ---- | ------------------ |
| 1    | 8                  |
| 4    | 2                  |
| 16   | 0.5                |
| 64   | 0.125              |

Rank büyüdükçe klasik LoRA'nın ölçeği hızlı küçülür. Bu bazen yüksek rank
kullanıldığında adapterin etkisini gereğinden fazla zayıflatır.

---

## 9. rsLoRA Neyi Değiştiriyor?

rsLoRA, LoRA'nın yapısını değiştirmez. Yine iki küçük tablo vardır, yine ana
model dondurulur. Sadece ölçek hesabı değişir.

Klasik LoRA:

```text
ölçek = alpha / rank
```

rsLoRA:

```text
ölçek = alpha / karekök(rank)
```

Aynı `alpha = 8` için:

| rank | klasik LoRA | rsLoRA |
| ---- | ----------- | ------ |
| 1    | 8           | 8      |
| 4    | 2           | 4      |
| 16   | 0.5         | 2      |
| 64   | 0.125       | 1      |

Sezgisi:

> Rank büyüdüğünde adapterin ham etkisi rank kadar hızlı büyümez. Bu yüzden
> rank'e bölmek bazen fazla serttir. Karekök(rank)'e bölmek daha dengeli olur.

Kodda rsLoRA ayrı bir sınıf değil; `lora.py` içinde ölçek satırı farklı
hesaplanır.

---

## 10. Kodda LoRA Tam Olarak Nerede?

Temel sınıf `lora.py` içindeki `LoRALinear` sınıfıdır. Bu sınıf, mevcut bir
`nn.Linear` katmanını sarar.

Normal katman:

```text
girdi -> eski Linear katmanı -> çıktı
```

LoRA'lı katman:

```text
girdi -> eski Linear katmanı -> eski cevap
girdi -> LoRA adapteri       -> küçük düzeltme

son cevap = eski cevap + küçük düzeltme
```

Kodda önce eski katman saklanır:

```python
self.base = base
```

Sonra eski katman dondurulur:

```python
for p in self.base.parameters():
    p.requires_grad = False
```

Ardından iki küçük adapter tablosu eklenir:

```python
self.lora_A = nn.Parameter(torch.empty(cfg.r, in_f))
self.lora_B = nn.Parameter(torch.zeros(out_f, cfg.r))
```

Burada:

```text
lora_A: daraltan küçük tablo
lora_B: tekrar genişleten küçük tablo
```

Forward kısmında kodun yaptığı iş şudur:

```python
base_output = self.base(x)
adapter_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
return base_output + self.scale * adapter_output
```

Buradaki `x`, PyTorch kodunda katmana gelen sayı paketinin adıdır. Yani yukarıda
"girdi" dediğimiz şeydir.

Şöyle okuyabilirsin:

```text
1. Eski model cevabını hesapla.
2. Aynı girdiden küçük LoRA düzeltmesini hesapla.
3. Düzeltmeyi ölçekle.
4. Eski cevabın üzerine ekle.
```

Kodda `@` işareti matris çarpımıdır. Bu satırı bilmek zorunda değilsin; pratik
anlamı şudur:

```text
girdi -> A ile daralt -> B ile genişlet -> küçük düzeltme
```

---

## 11. `inject.py` Ne Yapıyor?

`inject.py`, LoRA'nın modele takıldığı yerdir.

Bir modelin içinde çok sayıda `Linear` katmanı vardır. `inject.py` bu katmanları
gezer ve hedeflenenleri adapterli katmana çevirir.

Örneğin `qwen3` için varsayılan hedefler:

```text
q_proj, k_proj, v_proj, o_proj
```

Bunlar attention içindeki projeksiyon katmanlarıdır.

Akış:

1. Modeldeki `Linear` katmanlar bulunur.
2. İsmi hedef listesinde olanlar seçilir.
3. Seçilen katman `LoRALinear`, `DoRALinear`, `VeRALinear` veya PiSSA için
   LoRA tabanlı katmanla değiştirilir.
4. Bütün model dondurulur.
5. Sadece adapter parametreleri trainable yapılır.

Bu yüzden eğitimde optimizer büyük modeli değil, sadece küçük adapter
parametrelerini değiştirir.

---

## 12. `train.py` Ne Yapıyor?

`train.py` bu fikri küçük bir Türkçe isim üretme görevinde gösterir.

Base model bütün Türkçe isimler üzerinde eğitilir. Sonra LoRA ile şu yapılır:

```text
Base model aynı kalsın.
Sadece küçük adapter eğitilsin.
Model belirli bir harfle başlayan isimler üretmeye yönelsin.
```

Örneğin hedef harf `z` ise, eğitim verisi sadece `z` ile başlayan isimlerden
oluşur. Base model dondurulur. Adapter eğitilir.

Ana akış:

```text
1. base_qwen3.pt yüklenir.
2. Base modelden örnek isimler üretilir.
3. Adapter modele takılır.
4. Trainable parametre sayısı yazdırılır.
5. Sadece adapter eğitilir.
6. Adapter açıkken sonuç ölçülür.
7. Adapter kapatılınca base davranışı kontrol edilir.
8. Küçük adapter dosyası kaydedilir.
9. Merge sonrası çıktı aynı mı diye kontrol edilir.
```

Bu, LoRA'nın pratik karşılığıdır:

> Büyük modeli yeniden eğitmeden, küçük bir dosya ile davranışı değiştirmek.

---

## 13. Adapter Açık, Kapalı ve Merge Edilmiş Ne Demek?

### Adapter açık

Model eski cevabın üzerine LoRA düzeltmesini ekler.

```text
son cevap = eski cevap + küçük düzeltme
```

Bu, hedef göreve uyarlanmış davranıştır.

### Adapter kapalı

Model sadece eski cevabı kullanır.

```text
son cevap = eski cevap
```

LoRA, rsLoRA, DoRA ve VeRA için bu base modele geri dönmek gibidir. PiSSA'da
küçük bir istisna vardır; çünkü PiSSA base ağırlığı residual hale getirir.

### Merge edilmiş

Adapterin öğrendiği düzeltme ana ağırlığın içine kalıcı olarak katılır.

Günlük dille:

```text
Adapter ayrı bir parça olmaktan çıkar.
Etkisi base katmanın içine yazılır.
```

Bundan sonra inference sırasında ekstra adapter yolu çalıştırmak gerekmez.

---

## 14. LoRA, rsLoRA, DoRA, VeRA, PiSSA Farkları

Bu klasörde beş yöntem var:

| Yöntem | Basit açıklama                                              | Eğitilen şey                     |
| ------ | ----------------------------------------------------------- | -------------------------------- |
| LoRA   | Büyük katman sabit kalır, iki küçük tablo düzeltme üretir.  | `A`, `B`                         |
| rsLoRA | LoRA ile aynı, sadece adapter gücü daha dengeli ölçeklenir. | `A`, `B`                         |
| DoRA   | Ağırlığın yönünü ve uzunluğunu ayrı düşünür.                | `A`, `B`, ayrıca uzunluk vektörü |
| VeRA   | `A` ve `B` bile eğitilmez; rastgele sabit kalır.            | sadece iki küçük ölçek vektörü   |
| PiSSA  | LoRA gibi çalışır ama başlangıcı daha akıllıdır.            | `A`, `B`                         |

Hepsinin ortak noktası:

```text
Base modelin büyük ağırlıkları korunur.
Küçük bir adapter davranışı değiştirir.
```

---

## 15. DoRA'yı Basitçe Anlamak

DoRA, LoRA'ya şunu ekler:

> Bir ağırlık satırının yönünü ve uzunluğunu ayrı ayrı kontrol edelim.

Basit vektör örneği:

```text
[3, 4]
```

Bu vektörün uzunluğu `5`tir. Çünkü 3-4-5 üçgeni gibi düşünebilirsin.

LoRA bu vektörü örneğin şuna itebilir:

```text
[4.5, 4]
```

Bu durumda hem yön değişmiştir hem uzunluk değişmiştir.

DoRA ise şöyle düşünür:

```text
yönü LoRA değiştirsin
uzunluğu ayrı bir küçük parametre değiştirsin
```

Bu yüzden DoRA'da LoRA'nın `A` ve `B` parçalarına ek olarak bir de uzunluk
vektörü vardır. Bu vektör `dora_mag` adıyla kodda görülür.

---

## 16. VeRA'yı Basitçe Anlamak

VeRA daha da az parametre eğitmek ister.

LoRA'da:

```text
A ve B eğitilir.
```

VeRA'da:

```text
A ve B rastgele oluşturulur ve sabit bırakılır.
Sadece iki küçük ölçek vektörü eğitilir.
```

Bunu şöyle düşünebilirsin:

```text
Rastgele hazırlanmış bir yön bankası var.
VeRA sadece bu yönlerin sesini açıp kapatmayı öğreniyor.
```

Kodda:

- `vera_A` ve `vera_B`: sabit rastgele tablolar
- `vera_d`: ara kanalların ses ayarı
- `vera_b`: çıktı kanallarının ses ayarı

VeRA'nın adapter dosyası küçük olabilir; çünkü sabit rastgele tabloları kaydetmek
yerine aynı seed ile tekrar üretir.

---

## 17. PiSSA'yı Basitçe Anlamak

PiSSA, LoRA'nın çalışma şeklini baştan değiştirmez. Sadece daha akıllı bir
başlangıç yapar.

Klasik LoRA:

```text
Adapter başlangıçta neredeyse sıfır etkidedir.
Eğitimle faydalı yönleri bulmaya çalışır.
```

PiSSA:

```text
Ana ağırlık tablosunun en önemli yönlerini bulur.
Adapteri bu önemli yönlerden başlatır.
Kalan kısmı base ağırlıkta bırakır.
```

Burada kullanılan teknik SVD'dir. SVD'yi ayrıntılı bilmek zorunda değilsin.
Bu bağlamda şöyle düşünebilirsin:

```text
SVD = büyük tabloyu en etkili yönlerine ayırma yöntemi
```

Küçük örnek:

```text
Ana tablo:
[[2, 1],
 [1, 2]]
```

PiSSA bunu iki parçaya ayırır:

```text
önemli parça:
[[1.5, 1.5],
 [1.5, 1.5]]

kalan parça:
[[ 0.5, -0.5],
 [-0.5,  0.5]]
```

Bu iki parçayı toplarsan yine eski tabloyu elde edersin. PiSSA'nın fikri:

```text
Önemli parça adapterde başlasın.
Kalan parça base olarak kalsın.
```

Bu yüzden PiSSA bazen daha iyi bir başlangıç sağlar.

---

## 18. Bu Repoda Süreç Nasıl İşliyor?

Bu klasördeki akış:

1. `pretrain.py` base modeli bütün isimler üzerinde eğitir.
2. Base model `base_<model>.pt` olarak kaydedilir. Varsayılan: `base_qwen3.pt`.
3. `train.py` base modeli yükler.
4. `inject.py` seçilen adapteri hedef katmanlara takar.
5. Base model dondurulur.
6. Sadece adapter eğitilir.
7. Adapter dosyası kaydedilir.
8. `generate.py` base model + adapter ile isim üretir.

Ana dosyalar:

| Dosya         | Görev                                                 |
| ------------- | ----------------------------------------------------- |
| `lora.py`     | LoRA ve rsLoRA'nın temel katmanı                      |
| `dora.py`     | DoRA adapteri                                         |
| `vera.py`     | VeRA adapteri                                         |
| `pissa.py`    | PiSSA başlangıç mantığı                               |
| `inject.py`   | Adapterleri modele takma, dondurma, kaydetme, yükleme |
| `by_hand.py`  | Küçük sayılarla elle kontrol edilebilir örnekler      |
| `pretrain.py` | Base modeli eğitir                                    |
| `train.py`    | Adapteri eğitir                                       |
| `generate.py` | Kaydedilen adapter ile üretim yapar                   |

---

## 19. Nasıl Çalıştırılır?

`lora/` klasörüne geç:

```bash
cd lora
```

Önce elle kontrol edilebilir örnekleri çalıştır:

```bash
python3 by_hand.py
```

Base modeli eğit:

```bash
python3 pretrain.py
```

Varsayılan adapter eğitimi `qwen3 + lora + z` anlamına gelir:

```bash
python3 train.py
```

Aynı şeyi açık yazmak istersen:

```bash
python3 train.py qwen3 lora z
```

Diğer yöntemler:

```bash
python3 train.py qwen3 rslora z
python3 train.py qwen3 dora z
python3 train.py qwen3 vera z
python3 train.py qwen3 pissa z
```

Kaydedilen adapter ile isim üret:

```bash
python3 generate.py adapter_qwen3_lora_z.pt 20
```

Adapter kapalı halde karşılaştır:

```bash
python3 generate.py adapter_qwen3_lora_z.pt 20 base
```

---

## 20. Günlük Hayattan Benzetme

Base modeli büyük bir müzik sistemi gibi düşün.

Normalde bütün sistemi yeniden tasarlamak çok pahalıdır. LoRA ise sisteme küçük
bir ekolayzır takmak gibidir:

```text
Base model = ana ses sistemi
LoRA adapteri = küçük ekolayzır ayarı
```

Ana sistemi bozmadan sadece küçük ayarlarla sesi farklı hale getirirsin.

Bu repodaki isim üretme örneğinde de durum benzerdir:

```text
Base model: genel Türkçe isimler üretir.
LoRA adapteri: modeli örneğin z harfiyle başlayan isimlere yönlendirir.
```

---

## 21. Mini Alıştırmalar

### Alıştırma 1

Ana modelin eski davranışı şu olsun:

```text
Gelen iki sayıyı aynen geri ver.
```

Girdi:

```text
[2, 3]
```

Eski cevap:

```text
[2, 3]
```

### Alıştırma 2

LoRA adapteri şunu öğrensin:

```text
İlk sayıya bak.
Bu sayının 2 katını ilk çıktıya ekle.
Bu sayının 1 katını ikinci çıktıya ekle.
```

Girdi yine:

```text
[2, 3]
```

İlk sayı `2` olduğu için adapter düzeltmesi:

```text
[4, 2]
```

Son cevap:

```text
eski cevap      = [2, 3]
LoRA düzeltmesi = [4, 2]
son cevap       = [6, 5]
```

Bu küçük örnek LoRA'nın özünü gösterir:

```text
Modelin eski cevabı korunur.
Küçük adapter sadece düzeltme ekler.
```

---

## 22. En Kısa Özet

LoRA'nın hikayesi dört cümle:

1. Büyük model dondurulur.
2. Hedef katmanların yanına küçük adapterler takılır.
3. Eğitimde sadece bu küçük adapterler değişir.
4. Sonuçta küçük bir adapter dosyası, büyük modelin davranışını yeni göreve
   doğru yönlendirir.

Formül bilmeden akılda tutulacak en iyi cümle:

```text
LoRA = eski model cevabı + küçük ve eğitilebilir düzeltme
```
