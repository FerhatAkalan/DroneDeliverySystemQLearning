| Language | [English](#english)  | [Türkçe](#türkçe) |
|-----------|-------------|---------|

## <a name="english"></a>

# 🚁 Drone Delivery System with Q-Learning

<div align="center">
  <p><em>Intelligent Package Delivery System with Reinforcement Learning</em></p>
</div>

---

## 📖 About the Project

This project is an intelligent drone simulator that optimizes urban package deliveries using the **Reinforcement Learning (Q-Learning)** algorithm. Drones pick up packages from the cargo depot and learn the most efficient routes to deliver them to multiple delivery points.

### 🎯 Project Goals
- Simulate real-world logistics problems
- Demonstrate practical application of the Q-Learning algorithm
- Observe the learning process with an interactive visual interface
- Experiment with training parameters to achieve the best results

## ✨ Features

- 🤖 **Q-Learning Algorithm** with adjustable hyperparameters
- 🎮 **Interactive Simulation** with real-time animations  
- 🎯 **Manual & AI Modes** - Control drone or watch AI performance
- 💾 **Model Management** - Save/load trained Q-tables
- 🔋 **Battery System** with energy costs for different actions

### 📊 Simulation Details
- 🟢 **Cargo Depot**: Center where packages are picked up
- 🔴 **Delivery Points**: Randomly placed target locations (1-3)
- 🔵 **Drone**: Intelligent agent (battery, cargo status display)
- 🔋 **Battery Management**: Different energy costs for movement, takeoff, landing

### 💾 Model Management
- **Save/Load Q-Table**: Store trained models
- **Training Statistics**: Track reward and steps per episode
- **Speed Settings**: Control training and simulation speeds

## 🛠️ Technology Stack

- **Python 3.10+** - Main programming language
- **PyQt5** - GUI framework  
- **NumPy** - Numerical computations

## 🚀 Installation

```bash
# Python 3.10 or higher is required
python --version
# 1. Clone the Project
git clone https://github.com/FerhatAkalan/DroneDeliverySystemQLearning.git
cd DroneDeliverySystemQLearning
# 2. Install Dependencies
pip install PyQt5 numpy
# 3. Start the Simulator
python drone_delivery_system_q_learning.py
```

## 🎯 Usage

### 🎓 Model Training
1. **Set Parameters**:
   - Grid size (3x3 - 7x7)
   - Learning rate (Alpha): 0.01 - 1.0
   - Discount factor (Gamma): 0.1 - 0.999
   - Exploration rate (Epsilon): 0.1 - 1.0

2. **Select Training Mode**:
   - **Fast**: Fast training in visual interface
   - **Human**: Step-by-step visual tracking

3. **Start Training**: Click the "🚀 Start Training" button

### 🤖 AI Demo
1. After training, use the "🤖 Play with AI" button
2. Watch the trained drone's performance in real time
3. You can adjust the simulation speed

## 🧠 Q-Learning Algorithm

### 🔄 Action Space
| Action | Description | Condition |
|--------|-------------|-----------|
| `0` ⬇️ | Move down | Drone must be flying |
| `1` ➡️ | Move right | Drone must be flying |
| `2` ⬆️ | Move up | Drone must be flying |
| `3` ⬅️ | Move left | Drone must be flying |
| `4` 📦 | Pick/Drop cargo | Drone must be on the ground |
| `5` 🛫🛬 | Takeoff/Land | Always |

### 🏆 Reward System

#### ✅ Positive Rewards
- **Pick up cargo**: +50 points
- **Successful delivery**: +200 points
- **Task completion**: +200 + battery bonus
- **Approaching target**: +5 points
- **Correct action at correct position**: +10 points

#### ❌ Penalties
- **Invalid action**: -2 to -30 points
- **Battery depletion**: -100 points
- **Timeout**: -50 points

## 🎥 Project Video
https://github.com/user-attachments/assets/dc8752a9-d821-4c75-901b-993b0077d4b4

## 🔬 Experimental Results

### 📊 Training Performance
- **Grid Size**: 5x5
- **Training**: 50K episodes, 85%+ success rate

### 📈 Learning Curve
As training progresses, drone performance increases significantly:
- First 1000 episodes: Random behavior
- 1000-3000 episodes: Learning basic strategy
- 3000+ episodes: Optimized route planning

## 🤝 Contributing

If you want to contribute to this project:

1. **Fork** it
2. **Create a feature branch** (`git checkout -b feature/NewFeature`)
3. **Commit** (`git commit -am 'Add new feature'`)
4. **Push** (`git push origin feature/NewFeature`)
5. **Create a Pull Request**

### 💡 Contribution Areas

| 🚀 Feature | 📝 Description | 🎯 Difficulty |
|-----------|---------------|--------------|
| **Obstacle System** | Obstacles and wind effect in grid | 🟡 Medium |
| **Multi Drone** | Multiple drones at the same time | 🔴 Hard |
| **Deep Q-Learning** | Neural network-based learning | 🔴 Hard |
| **3D Visualization** | 3D environment | 🟡 Medium |
| **Real-Time Statistics** | Matplotlib integration | 🟢 Easy |
| **Sound Effects** | Drone sounds and feedback | 🟢 Easy |

---

<div align="center">
  <h3>⭐ If you like the project, don't forget to give a star! ⭐</h3>
  <p>If you have any questions, you can <a href="https://github.com/FerhatAkalan/DroneDeliverySystemQLearning/issues">open an issue</a>.</p>
  
  **🚁 Happy Coding! 🚁**
  
  <sub>Made with ❤️ by Ferhat Akalan</sub>
</div>

---

</br>

---

## <a name="türkçe"></a>

# 🚁 Q-Learning ile Drone Teslimat Sistemi

<div align="center">
  <p><em>Pekiştirmeli Öğrenme ile Akıllı Paket Teslimat Sistemi</em></p>
</div>

---

## 📖 Proje Hakkında

Bu proje, **Pekiştirmeli Öğrenme (Q-Learning)** algoritması kullanarak şehir içi paket teslimatlarını optimize eden akıllı bir dron simülatörüdür. Dronlar, kargo deposundan paketleri alıp birden fazla teslimat noktasına en verimli rotaları öğrenerek ulaştırırlar.

### 🎯 Proje Hedefleri
- Gerçek dünya lojistik problemlerini simüle etmek
- Q-Learning algoritmasının pratik uygulamasını göstermek  
- İnteraktif görsel arayüz ile öğrenme sürecini gözlemlemek
- Eğitim parametrelerini deneyimleyerek en iyi sonuçları elde etmek          

## ✨ Özellikler

- 🤖 **Q-Learning Algoritması** ayarlanabilir hiperparametrelerle
- 🎮 **İnteraktif Simülasyon** gerçek zamanlı animasyonlarla
- 🎯 **Manuel & AI Modları** - Drone kontrolü veya AI performansı izleme
- 💾 **Model Yönetimi** - Eğitilmiş Q-tablolarını kaydetme/yükleme
- 🔋 **Batarya Sistemi** farklı eylemler için enerji maliyetleri

### 📊 Simülasyon Detayları
- 🟢 **Kargo Deposu**: Paketlerin alındığı merkez
- 🔴 **Teslimat Noktaları**: Rastgele yerleştirilen hedef lokasyonlar (1-3 adet)
- 🔵 **Drone**: Akıllı ajan (batarya, kargo durumu gösterimi)
- 🔋 **Batarya Yönetimi**: Hareket, kalkış, iniş için farklı enerji maliyetleri

## 🛠️ Teknoloji Stack

- **Python 3.10+** - Ana programlama dili
- **PyQt5** - GUI framework
- **NumPy** - Sayısal hesaplamalar

## 🚀 Kurulum

```bash
# Python 3.10 veya üzeri gereklidir
python --version
# 1. Projeyi Klonlayın
git clone https://github.com/FerhatAkalan/DroneDeliverySystemQLearning.git
cd DroneDeliverySystemQLearning
# 2. Bağımlılıkları Yükleyin
pip install PyQt5 numpy
# 3. Simülatörü Başlatın
python drone_delivery_system_q_learning.py
```

## 🎯 Kullanım

### 🎓 Model Eğitimi
1. **Parametreleri Ayarlayın**:
   - Grid boyutu (3x3 - 7x7)
   - Learning rate (Alpha): 0.01 - 1.0
   - Discount factor (Gamma): 0.1 - 0.999
   - Exploration rate (Epsilon): 0.1 - 1.0

2. **Eğitim Modunu Seçin**:
   - **Fast**: Görsel arayüzde hızlı eğitim
   - **Human**: Adım adım görsel takip

3. **Eğitimi Başlatın**: "🚀 Eğitimi Başlat" butonuna tıklayın

### 🤖 AI Demo
1. Eğitim tamamlandıktan sonra "🤖 AI ile Oyna" butonunu kullanın
2. Eğitilmiş dronun performansını gerçek zamanlı izleyin
3. Simülasyon hızını ayarlayabilirsiniz

## 🧠 Q-Learning Algoritması

### 🔄 Eylem Uzayı
| Eylem | Açıklama | Koşul |
|-------|----------|-------|
| `0` ⬇️ | Aşağı hareket | Drone havada olmalı |
| `1` ➡️ | Sağa hareket | Drone havada olmalı |
| `2` ⬆️ | Yukarı hareket | Drone havada olmalı |
| `3` ⬅️ | Sola hareket | Drone havada olmalı |
| `4` 📦 | Kargo al/bırak | Drone yerde olmalı |
| `5` 🛫🛬 | Kalkış/İniş | Her zaman |

### 🏆 Ödül Sistemi

#### ✅ Pozitif Ödüller
- **Kargo alma**: +50 puan
- **Başarılı teslimat**: +200 puan
- **Görev tamamlama**: +200 + batarya bonusu
- **Hedefe yaklaşma**: +5 puan
- **Doğru konumda eylem**: +10 puan

#### ❌ Cezalar
- **Geçersiz eylem**: -2 ile -30 puan arası
- **Batarya bitimi**: -100 puan
- **Zaman aşımı**: -50 puan

## 🎥 Proje Videosu
https://github.com/user-attachments/assets/dc8752a9-d821-4c75-901b-993b0077d4b4

## 🔬 Deneysel Sonuçlar

### 📊 Eğitim Performansı
- **Grid Boyutu**: 5x5
- **Eğitim**: 50K episode, %85+ başarı oranı

### 📈 Öğrenme Eğrisi
Eğitim ilerledikçe dronun performansı belirgin şekilde artar:
- İlk 1000 episode: Rastgele davranış
- 1000-3000 episode: Temel strateji öğrenme
- 3000+ episode: Optimize edilmiş rota planlama

## 🤝 Katkıda Bulunma

Bu projeye katkıda bulunmak istiyorsanız:

1. **Fork** edin
2. **Feature branch** oluşturun (`git checkout -b feature/NewFeature`)
3. **Commit** edin (`git commit -am 'Add new feature'`)
4. **Push** edin (`git push origin feature/NewFeature`)
5. **Pull Request** oluşturun

### 💡 Katkı Alanları

| 🚀 Özellik | 📝 Açıklama | 🎯 Zorluk |    
|-----------|-------------|----------|    
| **Engel Sistemi** | Grid'e engeller ve rüzgar etkisi | 🟡 Orta |    
| **Çoklu Drone** | Aynı anda birden fazla drone | 🔴 Zor |    
| **Deep Q-Learning** | Neural network tabanlı öğrenme | 🔴 Zor |    
| **3D Görselleştirme** | 3D ortam | 🟡 Orta |    
| **Gerçek Zamanlı İstatistikler** | Matplotlib entegrasyonu | 🟢 Kolay |    
| **Ses Efektleri** | Drone sesleri ve geri bildirimler | 🟢 Kolay |
---

<div align="center">
  <h3>⭐ Projeyi beğendiyseniz star vermeyi unutmayın! ⭐</h3>
  <p>Herhangi bir sorunuz varsa <a href="https://github.com/FerhatAkalan/DroneDeliverySystemQLearning/issues">issue açabilirsiniz</a>.</p>
  
  **🚁 Happy Coding! 🚁**
  
  <sub>Made with ❤️ by Ferhat Akalan</sub>
</div>
