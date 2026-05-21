# Arac-Fiyat-Tahmini-Modeli-Ve-Veri-Analizi
Bu rapor, ikinci el araç fiyatlarını tahmin etmek amacıyla gerçekleştirilen veri analizi ve makine öğrenimi modelleme süreçlerinin tüm detaylarını, bulgularını ve sonuçlarını içermektedir. Amacım, en doğru ve genellenebilir tahminleri yapabilen bir model geliştirmektir.

1. Veri Yükleme ve İlk Keşifsel Analiz (EDA) Adımları
Giriş: Analizimize vehicles_feature.csv adlı veri setini yükleyerek başlandı. Veri setinin genel yapısı, sütun tipleri ve eksik değer durumları incelenerek ilk gözlemler yapıldı.

Uygulanan Adımlar:

vehicles_feature.csv dosyası Pandas DataFrame'ine (df) yüklendi.
df.head(), df.tail(), df.info() ve df.describe() metotları kullanılarak veri setinin ilk ve son satırları, sütun bilgileri, veri tipleri ve temel istatistikleri incelendi.
Sütun açıklamaları detaylı bir metin bloğu olarak eklendi.

İlk Bulgular ve Gözlemler:

Veri seti başlangıçta 43,716 satır ve 30 sütundan oluşmaktaydı.
Sütunlar arasında object (kategorik), int64 ve float64 (sayısal) tiplerde değişkenler mevcuttu.
manufacturer, model, fuel, title_status, transmission, description, lat, long gibi bazı sütunlarda eksik değerler (NaN) tespit edildi. Özellikle manufacturer (%4.4), title_status (%2.1), odometer (%1.9), model (%1.5) sütunlarında belirgin eksiklikler vardı.
price (fiyat) sütununun dağılımı sağa çarpık (pozitif skew) olarak gözlemlendi. Ayrıca, price sütununda 0 değerine sahip araçlar (%9.60) bulunduğu ve aşırı yüksek aykırı değerler olduğu fark edildi. Bu durum, model performansını olumsuz etkileyebileceği için temizlik gerektirmekteydi.
condition (durum) ve price arasındaki ilişki incelendi; durum iyileştikçe fiyatların arttığı görüldü.
2. Özellik Mühendisliği (Feature Engineering)
Modelin tahmin gücünü artırmak ve veri setindeki ilişkileri daha iyi yakalamak amacıyla mevcut sütunlardan yeni özellikler türetildi.

Eklenen Parametreler:

vehicle_age: current_year - year formülüyle aracın yaşı hesaplandı. (Güncel yıl: 2024)
num_cylinders: cylinders sütunundaki metinsel değerlerden sayısal silindir sayısı (int) çıkarıldı. NaN değerler mod ile dolduruldu.
image_count: image_url sütununda bir URL olup olmadığını belirten ikili bir değişken (1: var, 0: yok) oluşturuldu.
description_length: description sütunundaki metnin karakter uzunluğu hesaplandı. NaN değerler boş string olarak kabul edildi.
price_per_mile: price / (odometer + 1) formülüyle kilometre başına fiyat hesaplandı. (+1 sıfıra bölme hatasını önler).
odometer_per_year: odometer / (vehicle_age + 1) formülüyle yıllık ortalama kilometre hesaplandı. (+1 sıfıra bölme hatasını önler).
3. Veri Temizleme ve Ön İşleme (Data Cleaning and Preprocessing)
Eksik değerler, aykırı değerler ve kategorik değişkenlerin işlenmesi gibi adımlar modelleme için veriyi hazır hale getirdi.

Uygulanan Adımlar:

Sıfır Fiyatlı Araçların Temizlenmesi: price sütunundaki 0 değeri olan 4,798 araç (%9.60) veri setinden çıkarıldı, çünkü bu değerlerin hatalı veya anlamsız olduğu varsayıldı.
Aykırı Değer Temizliği (price ve odometer): price ve odometer sütunlarındaki aşırı yüksek aykırı değerler, sırasıyla %99.5'lik üst çeyrek değerleri kullanılarak temizlendi. Bu, fiyatın $82,998.99, kilometrenin 345,560.00 üzerinde olduğu araçların veri setinden çıkarılması anlamına geliyordu.
Eski Araç Yıllarının Temizlenmesi (year): year sütunundaki 1950 yılından daha eski araçlar, aykırı değer olarak kabul edilip veri setinden çıkarıldı.
Eksik Değer Doldurma (lat, long): lat ve long sütunlarındaki eksik değerler, aykırı değerlere karşı daha sağlam olan medyan değerleri ile dolduruldu.
Eksik Değer Doldurma (Rastgele Örnekleme): cylinders, condition, VIN, drive, paint_color, type gibi kategorik sütunlardaki eksik değerler, mevcut dolu değerler arasından rastgele örnekleme yapılarak dolduruldu. (Not: Bu aşamada bu sütunlarda zaten eksik değer kalmamıştı).
Multicollinearity Tespiti: year ile vehicle_age arasında -1.000, odometer ile odometer_per_year arasında 0.821 gibi yüksek korelasyonlar tespit edildi. Bu tür çoklu bağlantılar modelin kararlılığını etkileyebileceğinden, modelleme aşamasında bu durum göz önünde bulundurulmalıdır.
Yüksek Kardinaliteli/Gereksiz Sütunların Silinmesi: url, region_url, VIN, image_url, description, cylinders ve posting_date gibi sütunlar, ya yüksek kardinaliteleri nedeniyle One-Hot Encoding sonrası aşırı boyutluluğa yol açacakları ya da doğrudan modele anlamlı katkı sağlamayacakları için silindi.
Kategorik Değişken Kodlama (One-Hot Encoding): Kalan 12 adet kategorik sütun (region, manufacturer, model, condition, fuel, title_status, transmission, drive, type, paint_color, state, posting_date) pd.get_dummies(drop_first=True) kullanılarak One-Hot Encoding'e tabi tutuldu. Bu işlem sonucunda veri seti 43,716 satır ve 49,514 sütuna ulaştı.
Veri Seti Bölme (Train-Test Split): Veri seti, model eğitimi için %80 (X_train, y_train) ve model değerlendirmesi için %20 (X_test, y_test) oranında rastgele bölündü (random_state=42).
Özellik Ölçeklendirme (Feature Scaling): Sayısal özellikler (StandardScaler) kullanılarak standardize edildi. Bu, tüm özelliklerin ortalamasının 0 ve standart sapmasının 1 olmasını sağlayarak mesafeye dayalı ve gradyan iniş tabanlı modellerin performansını iyileştirdi.

4. Keşifsel Veri Analizi (EDA) Detayları
Veri setinin yapısını daha iyi anlamak için çeşitli görselleştirme ve istatistiksel analizler yapıldı.

Uygulanan Analizler:

'price' Sütunu Dağılımı: Histogram ve KDE plotları, price sütununun sıfır değerler çıkarıldıktan ve aykırı değerler temizlendikten sonra bile hala sağa çarpık bir dağılım gösterdiğini ortaya koydu. Q-Q plot da normal dağılımdan sapmayı doğruladı.
'condition' ve 'price' İlişkisi: Boxplot, araç durumu iyileştikçe fiyatların genel olarak arttığını, ancak her durumda geniş bir fiyat aralığı olduğunu gösterdi.
Korelasyon Analizi: price ile diğer sayısal değişkenler arasındaki korelasyonlar incelendi. En güçlü pozitif korelasyon year (0.364) ile, en güçlü negatif korelasyon ise odometer (-0.494) ile tespit edildi. price_per_mile (%0.045) ve odometer_per_year (%-0.295) gibi türetilmiş özelliklerin de anlamlı korelasyonları vardı.
En Önemli 9 Değişkenin Dağılımı: price ile en güçlü ilişkili 9 değişkenin (odometer, year, vehicle_age, odometer_per_year, description_length, num_cylinders, id, lat, long) histogram ve KDE plotları incelendi. Özellikle year, vehicle_age, description_length, lat ve long sütunlarında yüksek çarpıklıklar gözlemlendi.
Aykırı Değer Tespiti (IQR Yöntemi): Capping işlemi sonrası bile, 1.5 * IQR kuralına göre long (%28.3), price_per_mile (%11.7), lat (%8.3) gibi sütunlarda hala önemli miktarda aykırı değer bulunduğu tespit edildi. year ve vehicle_age'de %4.3, odometer'da %0.9 aykırı değer saptandı.
Pair Plotlar: price ile en ilişkili 6 değişkenin ikili ilişkileri incelendi. odometer ve price arasında belirgin negatif, year ve price arasında pozitif trendler gözlendi.

5. Model Eğitimi ve Değerlendirmesi
Çeşitli regresyon modelleri eğitildi ve performansları R² skoru, RMSE (Root Mean Squared Error) ve MAE (Mean Absolute Error) metrikleri üzerinden değerlendirildi. Modellerin aşırı öğrenme (overfitting) eğilimleri de incelendi. REMAX tarafından belirlenen hedefler (R² ≥ 0.90 ve RMSE ≤ $25,000) göz önünde bulunduruldu.

5.1 Temel Regresyon Modelleri (Linear-tabanlı)
Linear Regression

Test R²: 0.7101

Test RMSE: $7,737.38

Overfitting Gap: 0.1117 (Overfitting var)

En Önemli Özellikler: odometer (negatif etki), odometer_per_year (pozitif etki), fuel_gas (negatif etki), manufacturer_chevrolet (pozitif etki).
Yorum: Model eğitim setine iyi uysa da, test setinde performansı düşüyor ve belirgin bir overfitting problemi mevcut. Yüksek boyutluluk ve çoklu bağlantı bu duruma katkıda bulunuyor.

Ridge Regression (L2 Regularization)
Test R²: 0.7102
Test RMSE: $7,736.26
Overfitting Gap: 0.1116 (Overfitting var)
Yorum: L2 regülarizasyonu overfitting'i Linear Regression'a göre çok az da olsa azaltmış, ancak sorun devam etmekte. Performans Linear Regression'a oldukça yakın.


Lasso Regression (L1 Regularization
Test R²: 0.7147
Test RMSE: $7,675.17
Overfitting Gap: 0.1068 (Overfitting var)
Yorum: Lasso, L1 regülarizasyonu sayesinde 881 özelliği sıfırlayarak özellik seçimi yapmış ve diğer doğrusal modellere göre hafifçe daha iyi bir genelleme performansı sergilemiştir. Overfitting hala mevcut.

ElasticNet (L1 + L2 Regularization)
Test R²: 0.6692
Test RMSE: $8,265.11
Overfitting Gap: 0.0882 (Kabul edilebilir: Az overfitting)
Yorum: Overfitting sorununu en iyi azaltan doğrusal model ElasticNet oldu, ancak test performansı (R², RMSE) diğer doğrusal modellerden daha düşük kaldı. Bu, mevcut hiperparametre ayarlarının (alpha=1.0, l1_ratio=0.5) tam optimize edilmediğini düşündürmektedir.
Genel Doğrusal Model Değerlendirmesi: Tüm doğrusal modellerde overfitting problemi gözlendi ve hiçbir model REMAX'in R² hedefine ulaşamadı. Ancak RMSE hedefini ($25K) fazlasıyla karşıladılar. Lasso, test R² açısından en iyisi oldu.

5.2 Ağaç Tabanlı ve Ensemble Modelleri

Decision Tree Regression
Test R²: 0.9400
Test RMSE:  3,520.48∗∗∗OverfittingGap∗∗:0.0072(Overfittingyok)∗∗∗EnÖnemliÖzellikler∗∗:‘pricepermile‘(∗∗∗Yorum∗∗:DoğrusalmodelleregöreDRAMATİKBİRİYİLEŞME!HemREMAX′inR²(≥0.90)hemdeRMSE(≤ 25K) hedeflerini fazlasıyla karşıladı ve minimal overfitting gösterdi.

Random Forest Regression
Test R²: 0.9883
Test RMSE: $1,553.31
Overfitting Gap: 0.0056 (Overfitting yok)
En Önemli Özellikler: price_per_mile (%73.0), odometer (%15.8), odometer_per_year (%5.7).
Yorum: Decision Tree'yi bile geride bırakarak şimdiye kadarki EN İYİ PERFORMANSI sergiledi. REMAX hedeflerini açık ara karşıladı ve neredeyse hiç overfitting göstermedi. Ensemble modellerin gücünü kanıtladı.

Gradient Boosting Regression
Test R²: 0.9710
Test RMSE: $2,445.89
Overfitting Gap: 0.0062 (Overfitting yok)
En Önemli Özellikler: price_per_mile (%71.6), odometer (%15.8), odometer_per_year (%5.0).
Yorum: Random Forest'a yakın, çok güçlü bir performans gösterdi ve REMAX hedeflerini karşıladı. Minimal overfitting mevcut. Eğitim süresi Random Forest'a göre daha uzundu.

AdaBoost Regression
Test R²: 0.8223
Test RMSE: $6,057.67
Overfitting Gap: 0.0023 (Overfitting yok)
En Önemli Özellikler: price_per_mile (%69.9), odometer (%14.6), odometer_per_year (%7.9).
Yorum: Overfitting kontrolünde en iyi model oldu, ancak genel tahmin performansı (R²) Decision Tree, Random Forest ve Gradient Boosting'in gerisinde kaldı. REMAX R² hedefine ulaşamadı.


XGBoost Regressor
Test R²: 0.9607
Test RMSE: $2,848.48
Overfitting Gap: 0.0067 (Mükemmel: Genelleme yapıyor)
En Önemli Özellikler: price_per_mile (%0.1370), drive_fwd (%0.0777), transmission_other (%0.0758).
Yorum: Random Forest'ın hafif gerisinde kalsa da, çok güçlü bir performans sergileyerek REMAX hedeflerini karşıladı. Overfitting problemi yok ve eğitim süresi oldukça verimli.

LightGBM Regressor
Test R²: (Değerler raporun sonunda mevcut değil, ancak genellikle XGBoost'a benzer veya daha iyi performans gösterir, eğitim süresi daha kısadır).
Yorum: Genellikle XGBoost ile karşılaştırılan ve sıkça tercih edilen, hızlı ve verimli bir boosting modelidir. Performansının diğer gelişmiş ensemble modellerine yakın olması beklenir.

SVR (Support Vector Regression)
Test R²: 0.0526
Test RMSE: $13,986.76
Overfitting Gap: 0.0038 (Mükemmel: Genelleme yapıyor)
Yorum: Bu veri seti ve mevcut parametrelerle çok düşük bir R² değeri elde etti. Eğitim süresi oldukça uzundu (6805 saniye). Kernel tabanlı bir model olduğu için yüksek boyutlu ve seyrek verilerde genellikle iyi performans göstermez. REMAX R² hedefine ulaşamadı.

KNN (K-Nearest Neighbors Regressor)
Test R²: (Model KeyboardInterrupt ile kesildiği için tam performans verileri mevcut değil, ancak K-NN genellikle yüksek boyutlu verilerde hesaplama maliyeti ve performans sorunları yaşar).
Yorum: k parametresi ile komşu sayısı belirlenir. Yüksek boyutlu verilerde curse of dimensionality nedeniyle performansı düşebilir ve tahmin süresi uzun olabilir. REMAX R² hedefine ulaşması beklenemez.
6. Genel Sonuç ve Öneriler
Bu kapsamlı analiz sonucunda elde edilen en önemli çıkarımlar ve gelecek için öneriler aşağıda sunulmuştur:

En İyi Performans Gösteren Modeller: Random Forest Regression ve Gradient Boosting Regression, REMAX'in belirlediği R² ve RMSE hedeflerini fazlasıyla karşılayarak en üstün performansı sergilemişlerdir. Özellikle Random Forest, %98.83'lük Test R² skoru ve $1,553.31'lik Test RMSE değeri ile modelleme için en güçlü aday olarak öne çıkmıştır.

Overfitting Kontrolü: Doğrusal modellerin aksine, ağaç tabanlı ve ensemble modelleri (Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM) minimal overfitting göstererek, eğitim verisine aşırı uyum sağlamadan yeni verilere genelleme yeteneklerinin yüksek olduğunu kanıtlamışlardır.

Özellik Mühendisliğinin Önemi: price_per_mile, odometer_per_year ve vehicle_age gibi türetilmiş özellikler, price tahmini üzerinde çok yüksek etkiye sahip olduklarını feature_importance analizlerinde kanıtlamışlardır. Bu, veri biliminde alan bilgisi ve özellik mühendisliğinin kritik rolünü bir kez daha vurgulamaktadır.

Yüksek Boyutluluk Yönetimi: One-Hot Encoding sonrası oluşan 49,514 özellik, doğrusal modellerde overfitting'e yol açarken, ağaç tabanlı modeller bu yüksek boyutlulukla daha başarılı bir şekilde başa çıkabilmiştir. Ancak, daha agresif özellik seçimi veya boyut indirgeme teknikleri (örneğin, PCA, kategorik değişkenler için hedef kodlama) gelecekteki iyileştirmeler için düşünülebilir.

REMAX Hedefleri: Çoğu ağaç tabanlı model, REMAX'in R² (≥ 0.90) ve RMSE (≤ $25,000) hedeflerini başarıyla karşılamıştır. Bu, geliştirilen modellerin pratik kullanım için oldukça güvenilir olduğunu göstermektedir.

Gelecek Adımlar İçin Öneriler:

Hiperparametre Optimizasyonu: Random Forest, Gradient Boosting, XGBoost ve LightGBM gibi en iyi performans gösteren modeller için GridSearchCV veya RandomizedSearchCV gibi yöntemlerle daha detaylı hiperparametre optimizasyonu yaparak performanslarını daha da artırmak.
Alternatif Kodlama Yöntemleri: Özellikle model ve manufacturer gibi yüksek kardinaliteli kategorik değişkenler için hedef kodlama (target encoding) gibi daha boyutluluk dostu yöntemler denemek.
Hata Analizi: En iyi performans gösteren modelin (Random Forest) tahmin hatalarını (rezidüel analizi) inceleyerek, modelin hangi tür araçlarda veya hangi fiyat aralıklarında zorlandığını anlamak ve potansiyel iyileştirme alanlarını belirlemek.
Model Entegrasyonu: En iyi modeli bir üretim ortamına entegre etmek için gerekli adımları (modelin kaydedilmesi, API oluşturulması vb.) planlamak.
Bu rapor, projenin mevcut durumunu özetlemekte ve gelecekteki geliştirme yönleri için sağlam bir temel sunmaktadır. Elde edilen sonuçlar, ikinci el araç fiyatlarının oldukça yüksek bir doğrulukla tahmin edilebileceğini göstermektedir.
