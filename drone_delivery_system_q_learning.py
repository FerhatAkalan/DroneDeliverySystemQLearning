# -*- coding: utf-8 -*-
"""
Paket Dağıtım Dronları Simülatörü
Vize Projesi - Ferhat

Senaryo: Dronlar, şehir içi teslimatlarda birden çok noktaya en verimli şekilde paket götürmeli.
Pekiştirmeli Öğrenme ile Yaklaşım (Taxi-v3 benzeri):
- Grid tabanlı ortam (ör: 5x5)
- Q-Learning ile öğrenme
- PyQt5 arayüzü
- Eğitim ve simülasyon hızları ayarlanabilir
- Q-Table kaydet/yükle
- Batarya, kargo, teslimat noktaları, take off/landing animasyonları
"""

import sys
import os
import random
import pickle
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout, QFileDialog, QMessageBox, QComboBox, QSlider)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QIcon, QPixmap

# =====================
# Ortam (Environment) Sınıfı
# =====================
class DroneDeliveryEnv:
    """
    Grid tabanlı şehir ortamı (Taxi-v3 benzeri):
    - Yeşil: Kargo deposu
    - Kırmızı: Teslimat noktaları
    - Mavi: Drone
    - Batarya, kargo, teslimatlar, uçuş durumu
    """
    def __init__(self, grid_size=5, max_steps=100, n_deliveries=1):
        # Ortamın temel parametreleri: grid boyutu, maksimum adım sayısı, teslimat noktası sayısı
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_deliveries = n_deliveries
        # Eylem uzayı: 
        # 0: Aşağı, 1: Sağa, 2: Yukarı, 3: Sola, 4: Kargo Al/Bırak, 5: Kalk/İn
        self.action_space_n = 6  # Drone'un yapabileceği toplam eylem sayısı
        #4 sabit teslimat noktası (köşeler)
        self.fixed_delivery_points = [
            np.array([0, 0]),
            np.array([0, self.grid_size-1]),
            np.array([self.grid_size-1, 0]),
            np.array([self.grid_size-1, self.grid_size-1])
        ]
        # Batarya tüketim oranları (her hareket/kalkış/iniş için)
        self.move_battery_cost = 1  # Normal hareket başına batarya tüketimi
        self.takeoff_battery_cost = 5  # Kalkış için batarya tüketimi
        self.landing_battery_cost = 5  # İniş için batarya tüketimi
        
        # Ortamı başlat
        self.reset()

    def reset(self):
        # Ortamı başlangıç durumuna sıfırlar. Her yeni bölüm (episode) başında çağrılır.
        # Drone'u grid üzerinde rastgele bir konumda başlat (Taxi-v3 mantığı)
        self.drone_pos = np.array([
            random.randint(0, self.grid_size-1),
            random.randint(0, self.grid_size-1)
        ])
        # Kargo deposunun konumu sabit.
        self.cargo_depot_pos = np.array([self.grid_size-1, self.grid_size-1])
        # Teslimat noktası sayısını her episode'da 1-3 arası rastgele seç
        self.n_deliveries = random.randint(1, 3)
        # Kargo deposu köşesini hariç tutarak teslimat noktası seç (Taxi-v3 mantığı)
        # Teslimat noktaları, kargo deposu olmayan köşelerden rastgele seçilir.
        available_indices = [i for i in range(len(self.fixed_delivery_points)) if not np.array_equal(self.fixed_delivery_points[i], self.cargo_depot_pos)]
        chosen_indices = random.sample(available_indices, self.n_deliveries)
        self.delivery_points = [self.fixed_delivery_points[i].copy() for i in chosen_indices]
        self.delivery_indices = chosen_indices  # State için indexler
        # Drone'un başlangıç durumu: kargo yok, batarya dolu, adım sayısı sıfır, teslimatlar yapılmamış.
        self.has_cargo = False
        self.battery = 100
        self.steps = 0
        self.delivered = [False]*len(self.delivery_points)
        self.done = False # Bölümün bitip bitmediğini gösterir.
        self.is_flying = False # Drone'un uçuş durumu.
        self.landing_state = "landed" # İniş/kalkış animasyon durumu.
        self.landing_animation_step = 0 # İniş/kalkış animasyon adımı.
        self.last_reward = 0  # Son adımda alınan ödül
        self.total_reward = 0  # Toplam ödül (her episode başında sıfırlanır)
        return self.get_state() # Ortamın mevcut durumunu döndürür.

    def get_state(self):
        # Ortamın mevcut durumunu temsil eden bir tuple döndürür.
        # Bu durum, Q-tablosunda anahtar olarak kullanılır.
        x, y = self.drone_pos
        state = (x, y, int(self.has_cargo), int(self.is_flying))
        for d in self.delivered:
            state += (int(d),)
        # Batarya seviyesi, state'e DAHA HASSAS dahil edilir (0-10 arası discretize)
        battery_level = min(int(self.battery / 10), 10)
        state += (battery_level,)
        # Teslimat noktası indexlerini state'e ekle
        for idx in self.delivery_indices:
            state += (idx,)
        return state  # Artık hash ve mod yok, doğrudan tuple kullanılıyor.

    def step(self, action):
        """
        Drone'a verilen eylemi uygular ve ortamı bir adım ilerletir.
        Args:
            action (int):
                0: Aşağı, 1: Sağa, 2: Yukarı, 3: Sola
                4: Kargo Al/Bırak (Yerdeyken kargo al veya teslim et)
                5: Kalk/İn (Take off/landing, uçuş durumunu değiştirir)
        Returns:
            tuple: (next_state, reward, done, info)
                next_state: Yeni durumun hashlenmiş temsili
                reward: Bu adımda alınan ödül/ceza
                done: Senaryo tamamlandı mı?
                info: Ek bilgi (ör. neden bitti, hangi eylem yapıldı)
        Drone kargo almak ve bırakmak için landing durumunda olmalı. Havadayken kargo bırakılamaz alınamaz.
        """
        if self.done:
            return self.get_state(), 0, True, {"info": "Senaryo zaten tamamlanmış."}
        
        # Başlangıç durumu
        old_pos = self.drone_pos.copy()
        reward = 0
        info = {}
        action_emojis = {
            0: '⬇️',  # Aşağı
            1: '➡️',  # Sağa
            2: '⬆️',  # Yukarı
            3: '⬅️',  # Sola
            4: '📦',  # Kargo Al/Bırak
            5: '🛫/🛬',  # Kalk/İn
        }
        action_names = {
            0: 'Aşağı hareket',
            1: 'Sağa hareket',
            2: 'Yukarı hareket',
            3: 'Sola hareket',
            4: 'Kargo Al/Bırak',
            5: 'Kalk/İn',
        }
        # --- Eylem tipine göre ödül/ceza ---
        if action <= 3:  # Hareket eylemleri
            if not self.is_flying:
                reward -= 2
                info["action"] = f"{action_emojis[action]} {action_names[action]} (action={action}) | Drone yerdeyken hareket edemez! Önce kalkış yapın."
            else:
                if action == 0:
                    self.drone_pos[0] = min(self.drone_pos[0] + 1, self.grid_size - 1)
                elif action == 1:
                    self.drone_pos[1] = min(self.drone_pos[1] + 1, self.grid_size - 1)
                elif action == 2:
                    self.drone_pos[0] = max(self.drone_pos[0] - 1, 0)
                elif action == 3:
                    self.drone_pos[1] = max(self.drone_pos[1] - 1, 0)
                if np.array_equal(old_pos, self.drone_pos):
                    reward -= 5
                    info["action"] = f"{action_emojis[action]} {action_names[action]} (action={action}) | Hareket edilemedi."
                else:
                    reward -= 1
                    self.battery -= self.move_battery_cost
                    info["action"] = f"{action_emojis[action]} {action_names[action]} (action={action})"
        elif action == 4:  # Kargo Al/Bırak
            if self.is_flying:
                reward -= 10
                info["action"] = f"{action_emojis[action]} {action_names[action]} (action={action}) | Drone havadayken kargo alınamaz/bırakılamaz! Önce iniş yapın."
            else:
                if np.array_equal(self.drone_pos, self.cargo_depot_pos) and not self.has_cargo:
                    self.has_cargo = True
                    reward += 50
                    info["action"] = f"{action_emojis[action]} Kargo alındı (action={action})"
                elif self.has_cargo:
                    delivered_any = False
                    for i, delivery_point in enumerate(self.delivery_points):
                        if np.array_equal(self.drone_pos, delivery_point) and not self.delivered[i]:
                            self.delivered[i] = True
                            self.has_cargo = False
                            reward += 200
                            info["action"] = f"{action_emojis[action]} {i+1}. teslimat tamamlandı (action={action})"
                            delivered_any = True
                            break
                    if not delivered_any:
                        reward -= 30
                        info["action"] = f"{action_emojis[action]} Yanlış yerde teslimat (action={action})"
                else:
                    reward -= 30
                    info["action"] = f"{action_emojis[action]} Burada kargo alınamaz/bırakılamaz (action={action})"
        elif action == 5:  # Kalk/İn
            if not self.is_flying:
                self.is_flying = True
                self.landing_state = "taking_off"
                self.landing_animation_step = 0
                reward -= 3
                info["action"] = f"🛫 Kalkış (action={action})"
                self.battery -= self.takeoff_battery_cost
            else:
                self.is_flying = False
                self.landing_state = "landing"
                self.landing_animation_step = 0
                reward -= 3
                info["action"] = f"🛬 İniş (action={action})"
                self.battery -= self.landing_battery_cost

        # --- Hedefe yaklaşma/uzaklaşma ödül/ceza ---
        target_pos = None
        if not self.has_cargo and not all(self.delivered):
            target_pos = self.cargo_depot_pos
        elif self.has_cargo:
            min_dist = float('inf')
            for i, point in enumerate(self.delivery_points):
                if not self.delivered[i]:
                    dist = np.sum(np.abs(self.drone_pos - point))
                    if dist < min_dist:
                        min_dist = dist
                        target_pos = point
        if target_pos is not None:
            old_dist = np.sum(np.abs(old_pos - target_pos))
            new_dist = np.sum(np.abs(self.drone_pos - target_pos))
            if self.is_flying and new_dist < old_dist:
                reward += 5  # Hedefe yaklaşma ödülü
            elif self.is_flying and new_dist > old_dist:
                reward -= 2  # Hedeften uzaklaşma cezası
            if np.array_equal(self.drone_pos, target_pos):
                if not self.is_flying and action == 4:
                    reward += 10  # Doğru yerde doğru eylem bonusu
                elif self.is_flying and action == 5:
                    reward += 5  # Doğru yerde iniş bonusu

        # --- Batarya kontrolü ---
        if self.battery <= 0:
            reward -= 100  # Batarya biterse ağır ceza
            self.battery = 0
            self.done = True
            info["done_reason"] = "Batarya bitti"

        # --- Adım sınırı ---
        self.steps += 1
        if self.steps >= self.max_steps:
            reward -= 50  # Maksimum adım cezası
            self.done = True
            info["done_reason"] = "Maksimum adım sayısına ulaşıldı"

        # --- Tüm teslimatlar tamamlandıysa ---
        if all(self.delivered):
            remaining_battery_bonus = self.battery
            reward += 200 + remaining_battery_bonus  # Büyük ödül ve kalan batarya bonusu
            self.done = True
            info["done_reason"] = f"Tüm teslimatlar tamamlandı! Kalan batarya: %{self.battery}"

        # İniş/kalkış animasyon durumlarını güncelle
        # Bu adımlar, görsel arayüzde animasyonun düzgün çalışmasını sağlar.
        if self.landing_state == "taking_off":
            self.landing_animation_step += 1
            if self.landing_animation_step >= 3:  # 3 adımda tamamlanan kalkış animasyonu
                self.landing_state = "flying"
        elif self.landing_state == "landing":
            self.landing_animation_step += 1
            if self.landing_animation_step >= 3:  # 3 adımda tamamlanan iniş animasyonu
                self.landing_state = "landed"
        
        self.last_reward = reward  # Son ödül bilgisini güncelle
        self.total_reward += reward  # Toplam ödülü güncelle
        # Son aksiyon bilgisini ortamda sakla
        self.last_action_info = info.get("action", "-")
        return self.get_state(), reward, self.done, info # Yeni durum, ödül, bölüm durumu ve ek bilgiyi döndür.
# =====================
# Q-Learning Ajanı
# =====================
class QLearningAgent:
    """
    Q-Learning ajanı: Epsilon-greedy, Q-Table, deneyim havuzu
    """
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        # Q-Learning parametreleri ve Q-Table başlatma
        self.env = env # Ajanın etkileşimde bulunacağı ortam.
        self.alpha = alpha  # Öğrenme oranı (learning rate): Yeni bilginin ne kadar dikkate alınacağını belirler.
        self.gamma = gamma  # İskonto faktörü (discount factor): Gelecekteki ödüllerin bugünkü değerini belirler.
        self.epsilon = epsilon  # Keşif oranı (exploration rate): Ajanın ne sıklıkla rastgele eylem seçeceğini belirler.
        self.epsilon_decay = epsilon_decay  # Epsilon azalma oranı: Epsilon'un her bölüm sonunda ne kadar azalacağını belirler.
        self.min_epsilon = min_epsilon  # Minimum keşif oranı: Epsilon'un düşebileceği en düşük değer.
        self.q_table = {}  # Q-Tablosu (durum-aksiyon değerleri): Her durum-eylem çifti için beklenen ödülü saklar.
        self.experience_buffer = []  # Deneyim havuzu (replay buffer): Ajanın geçmiş deneyimlerini saklar.
        self.buffer_size = 1000 # Deneyim havuzunun maksimum boyutu.
        self.batch_size = 32 # Deneyim tekrarı sırasında kullanılacak örneklem boyutu.
        self.learn_interval = 4 # Kaç adımda bir deneyim tekrarı yapılacağı.
        self.step_counter = 0 # Adım sayacı.

    def get_q_value(self, state, action):
        # Belirli bir durum ve aksiyon için Q-değerini döndür
        # Eğer durum Q-tablosunda yoksa, o durum için tüm eylemlerin Q-değerlerini sıfır olarak başlatır.
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.env.action_space_n)
        return self.q_table[state][action]

    def select_action(self, state, training=True):
        # Epsilon-greedy aksiyon seçimi
        # Eğitim modunda ve rastgele bir sayı epsilon'dan küçükse, rastgele bir eylem seçilir (keşif).
        # Aksi takdirde, mevcut durum için en yüksek Q-değerine sahip eylem seçilir (sömürü).
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space_n)  # Rastgele aksiyon (keşif)
        else:
            if state not in self.q_table: # Eğer durum Q-tablosunda yoksa, başlat.
                self.q_table[state] = np.zeros(self.env.action_space_n)
            max_value = np.max(self.q_table[state]) # En yüksek Q-değerini bul.
            # En yüksek Q-değerine sahip birden fazla eylem varsa, aralarından rastgele birini seç.
            max_indices = np.where(self.q_table[state] == max_value)[0]
            return np.random.choice(max_indices)  # En iyi aksiyon (sömürü)

    def learn(self, state, action, reward, next_state, done):
        # Q-Table güncellemesi ve deneyim havuzuna ekleme
        # Bu fonksiyon, ajanın bir eylem gerçekleştirdikten sonra Q-tablosunu güncellemesini sağlar.
        self.add_experience(state, action, reward, next_state, done) # Deneyimi havuza ekle.
        if state not in self.q_table: # Durum Q-tablosunda yoksa başlat.
            self.q_table[state] = np.zeros(self.env.action_space_n)
        if next_state not in self.q_table: # Sonraki durum Q-tablosunda yoksa başlat.
            self.q_table[next_state] = np.zeros(self.env.action_space_n)
        
        current_q = self.q_table[state][action] # Mevcut Q-değeri.
        # Eğer bölüm bittiyse (done=True), gelecekteki maksimum Q-değeri 0 olur.
        # Aksi takdirde, sonraki durum için maksimum Q-değeri alınır.
        max_future_q = 0 if done else np.max(self.q_table[next_state])
        # Q-değeri güncelleme formülü (Bellman denklemi).
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q # Q-tablosunu güncelle.
        
        self.step_counter += 1
        # Deneyim tekrarını belirli aralıklarla uygula
        # Deneyim havuzu yeterince doluysa ve belirli bir adım aralığına ulaşıldıysa deneyim tekrarı yapılır.
        if self.step_counter % self.learn_interval == 0 and len(self.experience_buffer) >= self.batch_size:
            self.experience_replay()

    def add_experience(self, state, action, reward, next_state, done):
        # Deneyim havuzuna yeni deneyim ekle
        # Eğer deneyim havuzu doluysa, en eski deneyim silinir.
        if len(self.experience_buffer) >= self.buffer_size:
            self.experience_buffer.pop(0)
        self.experience_buffer.append((state, action, reward, next_state, done)) # Yeni deneyimi ekle.

    def experience_replay(self):
        # Deneyim havuzundan rastgele örneklerle öğrenme
        # Bu, ajanın geçmiş deneyimlerinden tekrar öğrenmesini sağlayarak öğrenmeyi daha stabil hale getirir.
        batch = random.sample(self.experience_buffer, self.batch_size) # Havuzdan rastgele bir batch seç.
        for state, action, reward, next_state, done in batch: # Seçilen her deneyim için Q-değerini güncelle.
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.env.action_space_n)
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(self.env.action_space_n)
            current_q = self.q_table[state][action]
            max_future_q = 0 if done else np.max(self.q_table[next_state])
            replay_alpha = self.alpha * 0.7 # Deneyim tekrarı için biraz daha düşük bir öğrenme oranı kullanılabilir.
            new_q = current_q + replay_alpha * (reward + self.gamma * max_future_q - current_q)
            self.q_table[state][action] = new_q

    def decay_epsilon(self):
        # Epsilon'u kademeli olarak azalt
        # Bu, ajanın zamanla daha fazla sömürü yapmasını ve daha az keşif yapmasını sağlar.
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename):
        # Q-Tablosunu dosyaya kaydet
        # Eğitimli modelin daha sonra kullanılabilmesi için Q-tablosu kaydedilir.
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        # Q-Tablosunu dosyadan yükle
        # Daha önce eğitilmiş bir modelin Q-tablosu yüklenir.
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

# =====================
# Eğitim Thread'i (PyQt5)
# =====================
class TrainingThread(QThread): # PyQt5 QThread sınıfından miras alır, böylece arayüz donmadan eğitim yapılabilir.
    progress = pyqtSignal(int, float, float, float)  # episode, reward, steps, epsilon -> Eğitim ilerlemesini bildiren sinyal.
    finished = pyqtSignal(list, list) # Eğitim bittiğinde ödül ve adım listelerini gönderen sinyal.
    state_update = pyqtSignal() # Ortam durumunun güncellenmesi gerektiğini bildiren sinyal (görsel arayüz için).
    def __init__(self, env, agent, episodes, update_interval=10, mode="fast", delay=0.1): # "ansi" -> "fast"
        super().__init__()
        self.env = env # Eğitim ortamı.
        self.agent = agent # Eğitilecek ajan.
        self.episodes = episodes # Toplam eğitim bölümü sayısı.
        self.running = True # Eğitimin devam edip etmediğini kontrol eden bayrak.
        self.update_interval = update_interval # fast modunda ne sıklıkta arayüzün güncelleneceği.
        self.mode = mode  # 'human' (canlı izleme) veya 'fast' (hızlı eğitim).
        self.delay = delay  # 'human' modunda adımlar arası gecikme (saniye).
    def run(self):
        # Eğitim döngüsü (her episode için)
        rewards_per_episode = [] # Her bölümdeki toplam ödülü saklar.
        steps_per_episode = [] # Her bölümdeki adım sayısını saklar.
        for episode in range(self.episodes):
            if not self.running: # Eğer durdurma sinyali geldiyse eğitimi sonlandır.
                break
            state = self.env.reset() # Ortamı sıfırla.
            total_reward = 0 # Bu bölümdeki toplam ödül.
            done = False # Bölümün bitip bitmediği.
            step_counter = 0 # Bu bölümdeki adım sayısı.
            self.state_update.emit() # Arayüzü güncelle.
            while not done and self.running: # Bölüm bitene kadar veya durdurma sinyali gelene kadar devam et.
                action = self.agent.select_action(state, training=True) # Ajan bir eylem seçer.
                next_state, reward, done, info = self.env.step(action) # Ortamda eylemi uygula.
                self.agent.learn(state, action, reward, next_state, done) # Ajan öğrenir.
                state = next_state # Durumu güncelle.
                total_reward += reward # Toplam ödülü güncelle.
                step_counter += 1
                if self.mode == "human": # Eğer 'human' modundaysa
                    self.state_update.emit() # Arayüzü her adımda güncelle.
                    QThread.msleep(int(self.delay * 1000)) # Belirlenen süre kadar bekle.
                elif self.mode == "fast" and step_counter % self.update_interval == 0: # Eğer 'fast' modundaysa ve belirli aralıklarla # "ansi" -> "fast"
                    self.state_update.emit() # Arayüzü güncelle.
            rewards_per_episode.append(total_reward) # Bölüm ödülünü listeye ekle.
            steps_per_episode.append(self.env.steps) # Bölüm adım sayısını listeye ekle.
            self.agent.decay_epsilon() # Epsilon değerini azalt.
            self.state_update.emit() # Arayüzü güncelle.
            self.progress.emit(episode+1, total_reward, self.env.steps, self.agent.epsilon) # İlerleme sinyalini gönder.
        self.finished.emit(rewards_per_episode, steps_per_episode) # Eğitim bitti sinyalini gönder.
    def stop(self):
        # Eğitimi durdurmak için kullanılır.
        self.running = False

# =====================
# Grid ve Bilgi Paneli (PyQt5)
# =====================
class GridWidget(QWidget): # Ortamın grid yapısını görselleştiren widget.
    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env # Görselleştirilecek ortam.
        self.cell_size = 80 # Her bir grid hücresinin piksel boyutu.
        self.setMinimumSize(env.grid_size * self.cell_size, env.grid_size * self.cell_size)
        # Renkler ve görsel ayarlar
        self.colors = {
            'background': Qt.white,
            'grid': Qt.lightGray,
            'drone': Qt.blue,
            'drone_landed': QColor(100, 100, 180), # İniş yapmış drone rengi.
            'cargo_depot': Qt.green, # Kargo deposu rengi.
            'delivery_point': Qt.red, # Teslimat noktası rengi.
            'cargo': Qt.green, # Kargo rengi.
            'shadow': QColor(100, 100, 100, 80) # Drone uçarkenki gölge rengi.
        }
    def paintEvent(self, event):
        # Grid ve tüm nesneleri çiz
        # Bu fonksiyon, widget her yeniden çizildiğinde çağrılır.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing) # Daha pürüzsüz çizimler için.
        painter.fillRect(self.rect(), self.colors['background']) # Arka planı boya.
        # Ortalamak için offset hesapla
        grid_pixel_size = self.env.grid_size * self.cell_size
        x_offset = (self.width() - grid_pixel_size) // 2
        y_offset = (self.height() - grid_pixel_size) // 2
        # Grid çizgileri
        painter.setPen(QPen(self.colors['grid'], 1))
        for i in range(self.env.grid_size + 1):
            painter.drawLine(x_offset, y_offset + i * self.cell_size, x_offset + self.env.grid_size * self.cell_size, y_offset + i * self.cell_size)
            painter.drawLine(x_offset + i * self.cell_size, y_offset, x_offset + i * self.cell_size, y_offset + self.env.grid_size * self.cell_size)
        # Kargo deposu çizimi
        depot_x = x_offset + self.env.cargo_depot_pos[1] * self.cell_size + self.cell_size // 2
        depot_y = y_offset + self.env.cargo_depot_pos[0] * self.cell_size + self.cell_size // 2
        painter.setBrush(QBrush(self.colors['cargo_depot']))
        painter.setPen(Qt.NoPen) # Kenar çizgisi olmasın.
        painter.drawEllipse(depot_x - self.cell_size // 3, depot_y - self.cell_size // 3, 2 * self.cell_size // 3, 2 * self.cell_size // 3)
        # Teslimat noktaları çizimi
        painter.setBrush(QBrush(self.colors['delivery_point']))
        for i, point in enumerate(self.env.delivery_points):
            if i < len(self.env.delivered) and not self.env.delivered[i]: # Henüz teslim edilmemişse çiz.
                x = x_offset + point[1] * self.cell_size + self.cell_size // 2
                y = y_offset + point[0] * self.cell_size + self.cell_size // 2
                painter.drawEllipse(x - self.cell_size // 4, y - self.cell_size // 4, self.cell_size // 2, self.cell_size // 2)
                painter.setPen(Qt.black) # Teslimat noktası numarasını yazmak için.
                painter.setFont(QFont('Arial', 10))
                painter.drawText(x - 5, y + 5, str(i + 1)) # Teslimat noktası numarasını yaz.
                painter.setPen(Qt.NoPen)
        # Drone çizimi
        drone_x = x_offset + self.env.drone_pos[1] * self.cell_size + self.cell_size // 2
        drone_y = y_offset + self.env.drone_pos[0] * self.cell_size + self.cell_size // 2
        if self.env.is_flying: # Drone uçuyorsa
            # Gölge efekti
            painter.setBrush(QBrush(self.colors['shadow']))
            painter.drawEllipse(drone_x - self.cell_size // 6, drone_y + self.cell_size // 4, self.cell_size // 3, self.cell_size // 8)
            height_offset = 0 # Yükseklik ofseti (animasyon için).
            if self.env.landing_state == "taking_off": # Kalkış animasyonu
                height_offset = -5 * self.env.landing_animation_step
            elif self.env.landing_state == "landing": # İniş animasyonu
                height_offset = -15 + 5 * self.env.landing_animation_step
            elif self.env.landing_state == "flying": # Normal uçuş
                height_offset = -15
            drone_y += height_offset # Drone'un dikey konumunu ayarla.
            painter.setBrush(QBrush(self.colors['drone'])) # Uçan drone rengi.
        else: # Drone yerdeyse
            painter.setBrush(QBrush(self.colors['drone_landed'])) # İniş yapmış drone rengi.
        # Drone gövdesi
        painter.drawEllipse(drone_x - self.cell_size // 4, drone_y - self.cell_size // 4, self.cell_size // 2, self.cell_size // 2)
        # Pervaneler
        propeller_size = self.cell_size // 8
        if self.env.is_flying: # Uçarken pervaneler daha büyük görünebilir.
            propeller_size = self.cell_size // 6
        painter.setBrush(QBrush(Qt.black)) # Pervane rengi.
        # Sol üst
        painter.drawEllipse(drone_x - propeller_size - propeller_size//2, drone_y - propeller_size - propeller_size//2, propeller_size, propeller_size)
        # Sağ üst
        painter.drawEllipse(drone_x + propeller_size - propeller_size//2, drone_y - propeller_size - propeller_size//2, propeller_size, propeller_size)
        # Sol alt
        painter.drawEllipse(drone_x - propeller_size - propeller_size//2, drone_y + propeller_size - propeller_size//2, propeller_size, propeller_size)
        # Sağ alt
        painter.drawEllipse(drone_x + propeller_size - propeller_size//2, drone_y + propeller_size - propeller_size//2, propeller_size, propeller_size)
        # Kargo çizimi
        if self.env.has_cargo: # Eğer drone kargo taşıyorsa
            painter.setBrush(QBrush(self.colors['cargo'])) # Kargo rengi.
            painter.drawRect(drone_x - self.cell_size // 8, drone_y - self.cell_size // 8, self.cell_size // 4, self.cell_size // 4)
        # Batarya göstergesi
        painter.setPen(Qt.black)
        painter.setFont(QFont('Arial', 10))
        painter.drawText(drone_x - 20, drone_y - 30, f"🔋: {self.env.battery}%") # Drone üzerinde batarya seviyesini göster.

class InfoPanelWidget(QWidget): # Ortam ve eğitim bilgilerini gösteren widget.
    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env # Bilgileri gösterilecek ortam.
        layout = QVBoxLayout()
        # Bilgi paneli için GroupBox
        info_group = QGroupBox("ℹ️ Durum Bilgileri")
        info_layout = QVBoxLayout()
        self.battery_label = QLabel() # Batarya bilgisi etiketi.
        self.cargo_label = QLabel() # Kargo durumu etiketi.
        self.delivery_label = QLabel() # Teslimat durumu etiketi.
        self.steps_label = QLabel() # Adım sayısı etiketi.
        self.reward_label = QLabel()  # Son ödül etiketi
        self.total_reward_label = QLabel()  # Toplam ödül etiketi
        self.last_action_label = QLabel()  # Son aksiyon etiketi
        self.training_progress_label = QLabel()  # Eğitim ilerlemesi etiketi
        info_layout.addWidget(self.battery_label)
        info_layout.addWidget(self.cargo_label)
        info_layout.addWidget(self.delivery_label)
        info_layout.addWidget(self.steps_label)
        info_layout.addWidget(self.reward_label)  # Son ödül panelde göster
        info_layout.addWidget(self.total_reward_label)  # Toplam ödül panelde göster
        info_layout.addWidget(self.last_action_label)  # Son aksiyon panelde göster
        info_layout.addWidget(self.training_progress_label)  # Durum bilgisine eklendi
        info_group.setLayout(info_layout)
        self.status_label = QLabel() # Genel durum mesajları için etiket.
        # Ana layout
        layout.addWidget(info_group)
        layout.addWidget(self.status_label)
        layout.addStretch() # Widget'ları yukarıya iter.
        self.setLayout(layout)
        self.update_info() # Bilgileri ilk kez güncelle.
    def update_info(self):
        # Paneldeki tüm bilgileri günceller.
        self.battery_label.setText(f"🔋 Batarya: %{self.env.battery}")
        # Kargo etiketi: taşınıyorsa kalın yeşil
        if self.env.has_cargo:
            self.cargo_label.setText("📦 Kargo: <span style='color:#1ca81c; font-weight:bold;'>Taşınıyor</span>")
            self.cargo_label.setTextFormat(Qt.RichText) # HTML formatında metin.
            self.cargo_label.setStyleSheet("")
        else:
            self.cargo_label.setText("📦 Kargo: Yok")
            self.cargo_label.setTextFormat(Qt.AutoText)
            self.cargo_label.setStyleSheet("")
        # Teslimat etiketi: teslim edilen sayı yeşil ve kalın
        delivered_count = sum(self.env.delivered) # Teslim edilen paket sayısı.
        total = len(self.env.delivery_points) # Toplam teslimat noktası sayısı.
        if delivered_count > 0:
            self.delivery_label.setText(f"🎯 Teslimatlar: <span style='color:#1ca81c; font-weight:bold;'>{delivered_count}</span>/{total}")
            self.delivery_label.setTextFormat(Qt.RichText)
        else:
            self.delivery_label.setText(f"🎯 Teslimatlar: 0/{total}")
            self.delivery_label.setTextFormat(Qt.AutoText)
        self.steps_label.setText(f"👣 Adım: {self.env.steps}")
        self.reward_label.setText(f"🏅 Son Ödül: {self.env.last_reward:.2f}")  # Son ödül gösterimi
        self.total_reward_label.setText(f"🥇 Toplam Ödül: {self.env.total_reward:.2f}")  # Toplam ödül gösterimi
        # Son aksiyon bilgisini grup kutusunda göster
        if hasattr(self.env, 'last_action_info') and self.env.last_action_info:
            self.last_action_label.setText(f"🔄 Son Aksiyon: {self.env.last_action_info}")
        else:
            self.last_action_label.setText("🔄 Son Aksiyon: -")
        # Eğitim ilerlemesi sadece eğitim sırasında gösterilecek, aksi halde gizle
        if not self.training_progress_label.text(): # Eğer eğitim ilerleme metni boşsa
            self.training_progress_label.setVisible(False) # Etiketi gizle.
        else:
            self.training_progress_label.setVisible(True) # Etiketi göster.
    def set_status(self, status):
        # Genel durum mesajını ayarlar.
        self.status_label.setText(status)
    def set_training_progress(self, episode, total_episodes, reward, steps, epsilon):
        # Eğitim ilerleme bilgisini ayarlar.
        self.training_progress_label.setText(f"📈 Episode: {episode}/{total_episodes} | Ödül: {reward:.2f} | Adım: {steps} | Epsilon: {epsilon:.4f}")
        self.training_progress_label.setVisible(True) # Etiketi görünür yap.
    def clear_training_progress(self):
        # Eğitim ilerleme bilgisini temizler ve gizler.
        self.training_progress_label.setText("")
        self.training_progress_label.setVisible(False)

# =====================
# Ana PyQt5 Arayüzü
# =====================
class DroneDeliverySimulator(QMainWindow): # Ana uygulama penceresi.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paket Dağıtım Dronları Simülatörü - Q-Learning") # Pencere başlığı.

        # Emoji ikonu oluşturma
        emoji = "🚁"
        pixmap = QPixmap(64, 64) # İkon boyutu
        pixmap.fill(Qt.transparent) # Şeffaf arka plan
        painter = QPainter(pixmap)
        font = QFont()
        font.setPointSize(48) # Emoji boyutu
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, emoji)
        painter.end()
        self.setWindowIcon(QIcon(pixmap))

        self.resize(1200, 700) # Pencere boyutu.
        self.grid_size = 5 # Başlangıç grid boyutu.
        self.env = DroneDeliveryEnv(grid_size=self.grid_size) # Ortamı oluştur.
        self.agent = QLearningAgent(self.env) # Ajanı oluştur.
        self.training_thread = None # Eğitim thread'i başlangıçta yok.
        self.sim_speed = 50  # AI ile oyna hız (ms).
        # --- Ana Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget) # Ana widget'ı ayarla.
        main_layout = QHBoxLayout(central_widget) # Ana layout (yatay).
        # --- Sol Panel: Parametreler ve Kontroller ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel) # Sol panel layout'u (dikey).
        # Grid boyutu
        grid_group = QGroupBox("🗺️ Grid Ayarları")
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid Boyutu:"))
        self.grid_size_spin = QSpinBox() # Grid boyutu için spin box.
        self.grid_size_spin.setRange(3, 7) # Min ve max grid boyutu.
        self.grid_size_spin.setValue(self.grid_size)
        self.grid_size_spin.valueChanged.connect(self.update_grid_size) # Değer değiştiğinde fonksiyon çağır.
        grid_layout.addWidget(self.grid_size_spin)
        grid_group.setLayout(grid_layout)
        left_layout.addWidget(grid_group)
        # Q-Learning parametreleri
        ql_group = QGroupBox("🤖 Q-Learning Parametreleri")
        ql_layout = QGridLayout() # Parametreleri grid içinde düzenle.
        ql_layout.addWidget(QLabel("Alpha (Öğrenme Oranı):"), 0, 0)
        self.alpha_spin = QDoubleSpinBox(); self.alpha_spin.setRange(0.01, 1.0); self.alpha_spin.setSingleStep(0.01); self.alpha_spin.setValue(self.agent.alpha)
        ql_layout.addWidget(self.alpha_spin, 0, 1)
        ql_layout.addWidget(QLabel("Gamma:"), 1, 0)
        self.gamma_spin = QDoubleSpinBox(); self.gamma_spin.setRange(0.1, 0.999); self.gamma_spin.setSingleStep(0.01); self.gamma_spin.setValue(self.agent.gamma)
        ql_layout.addWidget(self.gamma_spin, 1, 1)
        ql_layout.addWidget(QLabel("Epsilon (Keşif Oranı):"), 2, 0)
        self.epsilon_spin = QDoubleSpinBox(); self.epsilon_spin.setRange(0.1, 1.0); self.epsilon_spin.setSingleStep(0.1); self.epsilon_spin.setValue(self.agent.epsilon)
        ql_layout.addWidget(self.epsilon_spin, 2, 1)
        ql_layout.addWidget(QLabel("Epsilon Decay:"), 3, 0)
        self.epsilon_decay_spin = QDoubleSpinBox(); self.epsilon_decay_spin.setRange(0.9, 0.9999); self.epsilon_decay_spin.setSingleStep(0.001); self.epsilon_decay_spin.setValue(self.agent.epsilon_decay)
        ql_layout.addWidget(self.epsilon_decay_spin, 3, 1)
        ql_layout.addWidget(QLabel("Min Epsilon:"), 4, 0)
        self.min_epsilon_spin = QDoubleSpinBox(); self.min_epsilon_spin.setRange(0.01, 0.5); self.min_epsilon_spin.setSingleStep(0.01); self.min_epsilon_spin.setValue(self.agent.min_epsilon)
        ql_layout.addWidget(self.min_epsilon_spin, 4, 1)
        ql_layout.addWidget(QLabel("Eğitim Episodes:"), 5, 0)
        self.episodes_spin = QSpinBox(); self.episodes_spin.setRange(100, 100000); self.episodes_spin.setSingleStep(100); self.episodes_spin.setValue(5000) # Eğitim bölümü sayısı.
        ql_layout.addWidget(self.episodes_spin, 5, 1)
        ql_group.setLayout(ql_layout)
        left_layout.addWidget(ql_group)
        # Eğitim hızı (mod ve delay)
        speed_group = QGroupBox("⚡ Eğitim/Simülasyon Hızı")
        speed_layout = QGridLayout()
        speed_layout.addWidget(QLabel("Eğitim Modu:"), 0, 0)
        self.training_mode_combo = QComboBox(); 
        self.training_mode_combo.addItems(["Fast Mode", "Human Mode"]) # Eğitim modu seçenekleri.
        speed_layout.addWidget(self.training_mode_combo, 0, 1)
        speed_layout.addWidget(QLabel("Eğitim Hızı (ms):"), 1, 0) # Canlı mod için eğitim hızı.
        speed_slider_row = QHBoxLayout()
        speed_slider_row.addWidget(QLabel("🏎️"))
        self.training_speed_slider = QSlider(Qt.Horizontal); self.training_speed_slider.setRange(10, 1000); self.training_speed_slider.setValue(100) # Hız ayarı için slider.
        speed_slider_row.addWidget(self.training_speed_slider)
        speed_slider_row.addWidget(QLabel("🐢"))
        speed_layout.addLayout(speed_slider_row, 1, 1)
        speed_layout.addWidget(QLabel("Simülasyon Hızı (ms):"), 2, 0) # AI ile oynama hızı.
        sim_slider_row = QHBoxLayout()
        sim_slider_row.addWidget(QLabel("🏎️"))
        self.sim_speed_slider = QSlider(Qt.Horizontal); self.sim_speed_slider.setRange(10, 1000); self.sim_speed_slider.setValue(self.sim_speed)
        self.sim_speed_slider.valueChanged.connect(self.update_sim_speed) # Değer değiştiğinde fonksiyon çağır.
        sim_slider_row.addWidget(self.sim_speed_slider)
        sim_slider_row.addWidget(QLabel("🐢"))
        speed_layout.addLayout(sim_slider_row, 2, 1)
        speed_group.setLayout(speed_layout)
        left_layout.addWidget(speed_group)
        # Eğitim kontrolleri
        training_group = QGroupBox("🎓 Eğitim")
        training_layout = QVBoxLayout()
        self.train_button = QPushButton("🚀 Eğitimi Başlat"); self.train_button.clicked.connect(self.start_training) # Eğitimi başlat butonu.
        self.stop_button = QPushButton("⏹️ Eğitimi Durdur"); self.stop_button.clicked.connect(self.stop_training); self.stop_button.setEnabled(False) # Eğitimi durdur butonu (başlangıçta pasif).
        self.save_button = QPushButton("💾 Modeli Kaydet"); self.save_button.clicked.connect(self.save_model); self.save_button.setEnabled(False) # Modeli kaydet butonu (başlangıçta pasif).
        self.load_button = QPushButton("📂 Modeli Yükle"); self.load_button.clicked.connect(self.load_model) # Modeli yükle butonu.
        training_layout.addWidget(self.train_button)
        training_layout.addWidget(self.stop_button)
        training_layout.addWidget(self.save_button)
        training_layout.addWidget(self.load_button)
        training_group.setLayout(training_layout)
        left_layout.addWidget(training_group)
        # Oyun kontrolleri
        game_group = QGroupBox("🎮 Oyun Kontrolleri")
        game_layout = QVBoxLayout()
        self.ai_button = QPushButton("🤖 AI ile Oyna"); self.ai_button.clicked.connect(self.play_with_ai); self.ai_button.setEnabled(False) # AI ile oyna butonu (başlangıçta pasif).
        self.human_button = QPushButton("🧑‍💻 İnsan Modu (Manuel Oyna)"); self.human_button.clicked.connect(self.play_human_mode) # Manuel oynama butonu.
        self.stop_game_button = QPushButton("⏹️ Oyunu Durdur"); self.stop_game_button.clicked.connect(self.stop_game); self.stop_game_button.setEnabled(False) # Oyunu durdur butonu (başlangıçta pasif).
        self.reset_button = QPushButton("🔄 Sıfırla"); self.reset_button.clicked.connect(self.reset_env) # Ortamı sıfırla butonu.
        game_layout.addWidget(self.ai_button)
        game_layout.addWidget(self.human_button)
        game_layout.addWidget(self.stop_game_button)
        game_layout.addWidget(self.reset_button)
        game_group.setLayout(game_layout)
        left_layout.addWidget(game_group)
        left_layout.addStretch() # Sol paneli yukarı iter.
        # --- Sağ Panel: Grid ve Bilgi Paneli ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel) # Sağ panel layout'u (dikey).
        self.grid_widget = GridWidget(self.env) # Grid widget'ını oluştur.
        self.info_panel = InfoPanelWidget(self.env) # Bilgi paneli widget'ını oluştur.
        right_layout.addWidget(self.grid_widget, 7) # Grid widget'ını ekle (daha fazla yer kaplasın).
        right_layout.addWidget(self.info_panel, 3) # Bilgi panelini ekle.
        # --- Layoutları birleştir ---
        main_layout.addWidget(left_panel, 1) # Sol paneli ana layout'a ekle (daha az yer kaplasın).
        main_layout.addWidget(right_panel, 4) # Sağ paneli ana layout'a ekle (daha fazla yer kaplasın).
        # --- Timer ---
        self.game_timer = QTimer(); self.game_timer.timeout.connect(self.update_game) # Oyun döngüsü için timer.
        self.game_mode = None # Oyun modu (ai, human, None).
        self.model_trained = False # Modelin eğitilip eğitilmediği.
        self.model_loaded = False # Modelin yüklenip yüklenmediği.
        self.update_ui() # Arayüzü ilk kez güncelle.
        self.statusBar().showMessage("Hazır - Eğitim veya AI ile oynamak için modeli eğitin/yükleyin.") # Durum çubuğu mesajı.

        # GitHub linki
        github_link_label = QLabel('Coded by <a href="https://github.com/FerhatAkalan">Ferhat Akalan</a>')
        github_link_label.setOpenExternalLinks(True)
        self.statusBar().addPermanentWidget(github_link_label)

    def set_params_enabled(self, enabled: bool):
        # Q-Learning ve grid parametrelerinin aktif/pasif durumunu ayarlar.
        self.grid_size_spin.setEnabled(enabled)
        self.alpha_spin.setEnabled(enabled)
        self.gamma_spin.setEnabled(enabled)
        self.epsilon_spin.setEnabled(enabled)
        self.epsilon_decay_spin.setEnabled(enabled)
        self.min_epsilon_spin.setEnabled(enabled)
        self.episodes_spin.setEnabled(enabled)
        self.training_mode_combo.setEnabled(enabled)
        self.training_speed_slider.setEnabled(enabled)
        self.sim_speed_slider.setEnabled(enabled)

    def set_game_buttons_enabled(self, enabled: bool):
        # Oyunla ilgili butonların aktif/pasif durumunu ayarlar.
        self.train_button.setEnabled(enabled)
        self.ai_button.setEnabled(enabled and (self.model_trained or self.model_loaded)) # AI butonu model varsa aktif olur.
        self.human_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled and self.model_trained) # Kaydet butonu model eğitildiyse aktif olur.
        self.load_button.setEnabled(enabled)

    def update_grid_size(self):
        # Grid boyutu değiştiğinde çağrılır.
        self.grid_size = self.grid_size_spin.value()
        self.reset_env() # Ortamı yeni grid boyutuyla sıfırla.
    def update_sim_speed(self):
        # Simülasyon hızı (AI ile oynama hızı) değiştiğinde çağrılır.
        self.sim_speed = self.sim_speed_slider.value()
        if self.game_timer.isActive(): # Eğer oyun zamanlayıcısı aktifse
            self.game_timer.setInterval(self.sim_speed) # Zamanlayıcının aralığını güncelle.
    def reset_env(self):
        # Ortamı ve ajanı sıfırlar.
        if self.game_timer.isActive(): # Eğer oyun zamanlayıcısı aktifse durdur.
            self.game_timer.stop(); self.game_mode = None
        self.env = DroneDeliveryEnv(grid_size=self.grid_size) # Yeni ortam oluştur.
        self.agent = QLearningAgent(self.env) # Yeni ajan oluştur (Q-tablosu sıfırlanır).
        self.grid_widget.env = self.env # Grid widget'ının ortamını güncelle.
        self.info_panel.env = self.env # Bilgi panelinin ortamını güncelle.
        self.model_trained = False # Model eğitilmedi olarak işaretle.
        self.model_loaded = False # Model yüklenmedi olarak işaretle.
        self.update_ui() # Arayüzü güncelle.
        self.set_game_buttons_enabled(True) # Butonları aktif et.
        self.set_params_enabled(True) # Parametreleri aktif et.
        self.info_panel.clear_training_progress() # Eğitim ilerlemesini temizle.
        self.statusBar().showMessage("Ortam sıfırlandı")
    def update_ui(self):
        # Arayüzdeki grid ve bilgi panelini günceller.
        self.grid_widget.update(); self.info_panel.update_info()
    def update_game(self):
        # AI ile oynama modunda oyunun bir adımını günceller.
        if self.game_mode == 'ai':
            if not hasattr(self, 'ai_episode_count'):
                self.ai_episode_count = 1
                self.ai_total_reward = 0
            if not hasattr(self, 'ai_episode_running'):
                self.ai_episode_running = True
            if not self.ai_episode_running:
                return  # Yeni episode başlatılana kadar bekle
            state = self.env.get_state()
            action = self.agent.select_action(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            self.ai_total_reward += reward
            self.update_ui()
            if done:
                delivered = sum(self.env.delivered)
                # Bilgi panelinde ve durum çubuğunda bölüm sonucunu göster.
                self.info_panel.set_status(f"🤖AI Episode: {self.ai_episode_count} | 🎯Teslimat: {delivered}/{len(self.env.delivery_points)} | 🔋Batarya: %{self.env.battery} | 🥇Skor: {self.ai_total_reward:.2f}")
                self.ai_episode_running = False  # Episode bitti, yeni episode başlatılana kadar adım atma
                self.ai_episode_count += 1
                self.ai_total_reward = 0
                QTimer.singleShot(200, self.env.reset)
                QTimer.singleShot(350, self.start_next_ai_episode)
                QTimer.singleShot(400, self.update_ui)

    def start_next_ai_episode(self):
        # Yeni AI episode başlatmak için flag'i tekrar aktif et
        self.ai_episode_running = True

    def start_training(self):
        # Eğitimi başlatır.
        # Ajan parametrelerini arayüzdeki değerlerle günceller.
        self.agent.alpha = self.alpha_spin.value()
        self.agent.gamma = self.gamma_spin.value()
        self.agent.epsilon = self.epsilon_spin.value()
        self.agent.epsilon_decay = self.epsilon_decay_spin.value()
        self.agent.min_epsilon = self.min_epsilon_spin.value()
        episodes = self.episodes_spin.value() # Eğitim bölümü sayısını al.
        mode_text = self.training_mode_combo.currentText() # Seçilen eğitim modunu al.
        training_mode = "human" if "human" in mode_text.lower() else "ansi" # Eğitim modunu belirle.
        delay = self.training_speed_slider.value() / 1000.0 if hasattr(self, 'training_speed_slider') else 0.1 # Canlı mod için gecikme.
        
        self.info_panel.clear_training_progress()  # Eğitim başında ilerleme bilgisini temizle.
        # Buton ve parametrelerin durumunu ayarla (eğitim sırasında çoğu pasif olur).
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_game_button.setEnabled(False)
        self.set_game_buttons_enabled(False)
        self.set_params_enabled(False)
        # ---
        self.env.reset() # Ortamı sıfırla.
        self.training_rewards = []; self.training_steps = [] # Ödül ve adım listelerini sıfırla.
        # Eğitim thread'ini oluştur ve başlat.
        self.training_thread = TrainingThread(self.env, self.agent, episodes, mode=training_mode, delay=delay)
        self.training_thread.progress.connect(self.update_training_progress) # İlerleme sinyaline bağlan.
        self.training_thread.finished.connect(self.training_finished) # Bitiş sinyaline bağlan.
        self.training_thread.state_update.connect(self.update_training_visualization) # Durum güncelleme sinyaline bağlan.
        self.training_thread.start() # Thread'i başlat.
        self.info_panel.set_status("Eğitim devam ediyor...")
        self.statusBar().showMessage(f"Eğitim başladı. Toplam episode: {episodes}")
    def update_training_visualization(self):
        # Eğitim sırasında arayüzü günceller (özellikle canlı modda).
        self.update_ui()
    def update_training_progress(self, episode, reward, steps, epsilon):
        # Eğitim ilerlemesini alır ve arayüzde gösterir.
        self.training_rewards.append(reward)
        self.training_steps.append(steps)
        # Eğitim bilgisi sadece eğitim sırasında gösterilecek
        self.info_panel.set_training_progress(episode, self.episodes_spin.value(), reward, steps, epsilon)
        self.info_panel.update_info() # Bilgi panelini güncelle.
        self.update_ui() # Genel arayüzü güncelle.
    def training_finished(self, rewards, steps):
        # Eğitim bittiğinde çağrılır.
        # Buton ve parametrelerin durumunu eski haline getirir.
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_game_button.setEnabled(False)
        self.set_game_buttons_enabled(True)
        self.info_panel.clear_training_progress()  # Eğitim bitince eğitim bilgisini gizle.
        # Son 100 bölümün ortalama ödül ve adım sayısını hesapla.
        avg_reward = sum(rewards[-100:]) / min(100, len(rewards)) if rewards else 0
        avg_steps = sum(steps[-100:]) / min(100, len(steps)) if steps else 0
        result_message = f"Eğitim tamamlandı!\n\nToplam episode: {len(rewards)}\nSon 100 episode ortalama ödül: {avg_reward:.2f}\nSon 100 episode ortalama adım: {avg_steps:.2f}\n\nŞimdi 'AI ile Oyna' butonunu kullanarak eğitilen modeli test edebilirsiniz."
        QMessageBox.information(self, "Eğitim Tamamlandı", result_message) # Bilgilendirme mesajı göster.
        self.statusBar().showMessage(f"Eğitim tamamlandı! Son 100 episode ortalama ödül: {avg_reward:.2f}, adım: {avg_steps:.2f}")
        self.training_thread = None; self.model_trained = True # Model eğitildi olarak işaretle.
        # self.training_status_label = QLabel("Model Durumu: Eğitildi"); self.training_status_label.setStyleSheet("color: green; font-weight: bold;") # Bu satır GUI'de bir yere eklenmeli.
        # Eğitim bitince parametreleri tekrar aktif et
        self.set_params_enabled(True)
        self.save_button.setEnabled(True) # Model eğitildiği için kaydet butonu aktif olur.
        self.ai_button.setEnabled(True) # Model eğitildiği için AI ile oyna butonu aktif olur.
    def stop_training(self):
        # Eğitimi durdurur.
        if self.training_thread:
            self.training_thread.stop() # Eğitim thread'ine durma sinyali gönder.
            self.statusBar().showMessage("Eğitim durduruluyor...")
            self.set_params_enabled(True) # Parametreleri tekrar aktif et.
            # Butonları da uygun şekilde ayarlamak gerekebilir.
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.set_game_buttons_enabled(True)
    def save_model(self):
        # Eğitilmiş Q-tablosunu kaydeder.
        save_dir = "models" # Kayıt dizini.
        if not os.path.exists(save_dir): os.makedirs(save_dir) # Dizin yoksa oluştur.
        # Dosya adı için zaman damgası ve grid boyutu kullanılır.
        timestamp = "qtable_" + str(self.grid_size) + "_" + str(random.randint(1000,9999)) + ".pkl"
        filename, _ = QFileDialog.getSaveFileName(self, "Q Tablosunu Kaydet", os.path.join(save_dir, timestamp), "Pickle Files (*.pkl);;All Files (*)") # Kayıt dialoğu.
        if filename: # Eğer bir dosya adı seçildiyse
            self.agent.save_q_table(filename) # Ajanın Q-tablosunu kaydet.
            self.statusBar().showMessage(f"Q tablosu kaydedildi: {filename}")
    def load_model(self):
        # Kaydedilmiş bir Q-tablosunu yükler.
        filename, _ = QFileDialog.getOpenFileName(self, "Q Tablosu Yükle", "models" if os.path.exists("models") else ".", "Pickle Files (*.pkl);;All Files (*)") # Yükleme dialoğu.
        if filename: # Eğer bir dosya seçildiyse
            try:
                self.agent.load_q_table(filename) # Ajanın Q-tablosunu yükle.
                self.model_loaded = True # Model yüklendi olarak işaretle.
                self.model_trained = True # Yüklenen model eğitilmiş sayılır.
                self.ai_button.setEnabled(True) # AI ile oyna butonunu aktif et.
                self.save_button.setEnabled(True) # Yüklenen model kaydedilebilir.
                self.statusBar().showMessage(f"Model başarıyla yüklendi: {filename}")
                # Yüklenen modelin parametrelerini arayüze yansıtmak iyi bir fikir olabilir (eğer saklanıyorsa).
                # Örneğin, grid boyutu modelle uyumlu olmalı.
                # Bu örnekte Q-tablosu grid boyutundan bağımsız değil, bu yüzden grid boyutu da modelle birlikte düşünülmeli.
                # Eğer model farklı bir grid boyutu için eğitilmişse, kullanıcıya bilgi verilebilir veya grid boyutu otomatik ayarlanabilir.
                # Şimdilik, yüklenen modelin mevcut grid boyutuyla uyumlu olduğunu varsayıyoruz.
            except Exception as e:
                QMessageBox.critical(self, "Model Yükleme Hatası", f"Model yüklenirken bir hata oluştu: {e}")
                self.model_loaded = False
                self.model_trained = False
                self.ai_button.setEnabled(False)
                self.save_button.setEnabled(False)

    def play_with_ai(self):
        # Eğitilmiş veya yüklenmiş model ile AI'ın oynamasını başlatır.
        if self.game_timer.isActive(): # Eğer zamanlayıcı zaten aktifse (yani AI oynuyorsa)
            self.stop_game(); return # Oyunu durdur.
        if not self.model_trained and not self.model_loaded: # Eğer model yoksa
            QMessageBox.warning(self, "Model Yok", "AI ile oynamak için önce modeli eğitmeniz veya yüklemeniz gerekiyor.")
            return
        self.env.reset(); self.game_mode = 'ai'; self.info_panel.set_status("AI ile oynanıyor...")
        self.info_panel.clear_training_progress()  # AI ile oyna başlarken eğitim bilgisini gizle.
        self.ai_episode_count = 1; self.ai_total_reward = 0 # AI bölüm sayacını ve ödülünü sıfırla.
        self.game_timer.start(self.sim_speed) # Oyun zamanlayıcısını başlat.
        # Buton ve parametrelerin durumunu ayarla.
        self.stop_game_button.setEnabled(True)
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.set_game_buttons_enabled(False) # Diğer oyun butonlarını pasif yap.
        self.ai_button.setEnabled(False) # AI ile oyna butonu zaten basıldığı için pasif.
        self.human_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.set_params_enabled(False) # Parametreleri pasif yap.

    def play_human_mode(self):
        # Kullanıcının manuel olarak oynamasını sağlar.
        if self.game_timer.isActive(): # Eğer AI oynuyorsa
            self.stop_game(); return # Oyunu durdur.
        self.env.reset(); self.game_mode = 'human'; # Ortamı sıfırla ve oyun modunu 'human' yap.
        self.info_panel.set_status("Manuel mod: Hareket=WASD/Ok Tuşları, Uç/Kalk/İn=Space, Kargo=E"); # Kullanıcıya bilgi ver.
        self.info_panel.clear_training_progress()  # İnsan modunda eğitim bilgisini gizle.
        self.update_ui() # Arayüzü güncelle.
        # Buton ve parametrelerin durumunu ayarla.
        self.stop_game_button.setEnabled(True)
        self.set_game_buttons_enabled(False) # Diğer oyun butonlarını pasif yap.
        self.train_button.setEnabled(False)
        self.ai_button.setEnabled(False)
        self.human_button.setEnabled(False) # Manuel mod butonu zaten basıldığı için pasif.
        self.reset_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.set_params_enabled(False) # Parametreleri pasif yap.
        self.setFocus() # Klavye girdilerini almak için pencereye odaklan.

    def keyPressEvent(self, event):
        # Klavye tuşlarına basıldığında çağrılır (sadece manuel modda).
        if self.game_mode != 'human' or self.env.done: # Eğer manuel modda değilse veya bölüm bittiyse bir şey yapma.
            return
        key = event.key() # Basılan tuşu al.
        action = None # Başlangıçta eylem yok.
        # WASD ve ok tuşları ile hareket
        if key in (Qt.Key_Down, Qt.Key_S): # Aşağı
            action = 0
        elif key in (Qt.Key_Right, Qt.Key_D): # Sağa
            action = 1
        elif key in (Qt.Key_Up, Qt.Key_W): # Yukarı
            action = 2
        elif key in (Qt.Key_Left, Qt.Key_A): # Sola
            action = 3
        # Kargo al/bırak: E
        elif key == Qt.Key_E:
            action = 4
        # Kalk/İn: Space
        elif key == Qt.Key_Space:
            action = 5
        
        if action is not None: # Eğer geçerli bir eylem tuşuna basıldıysa
            _, _, done, info = self.env.step(action) # Eylemi uygula.
            self.update_ui() # Arayüzü güncelle.
            if "action" in info and info["action"]: # Eğer eylemle ilgili bir mesaj varsa durum çubuğunda göster.
                self.statusBar().showMessage(info["action"])
            if done: # Eğer bölüm bittiyse
                self.info_panel.set_status("Oyun bitti! Manuel modda yeni oyun için 'Sıfırla' veya 'Oyunu Durdur' kullanın.")
                # Oyun bittiğinde bazı butonları tekrar aktif hale getirebiliriz.
                self.stop_game_button.setEnabled(False) # Oyunu durdur butonu pasif.
                self.reset_button.setEnabled(True) # Sıfırla butonu aktif.
                self.human_button.setEnabled(True) # Tekrar manuel oynamak için.
                # Diğer butonlar da duruma göre ayarlanabilir.

    def stop_game(self):
        # AI veya manuel oyunu durdurur.
        if self.game_timer.isActive() or self.game_mode == 'human': # Eğer AI oynuyorsa veya manuel moddaysa
            self.game_timer.stop(); self.game_mode = None # Zamanlayıcıyı durdur ve oyun modunu sıfırla.
            self.info_panel.set_status("Oyun durduruldu.")
            self.info_panel.clear_training_progress()  # Oyun durunca eğitim bilgisini gizle.
            self.statusBar().showMessage("Oyun durduruldu.")
            # Buton ve parametrelerin durumunu eski haline getir.
            self.stop_game_button.setEnabled(False)
            self.stop_button.setEnabled(False) # Eğitim durdurma butonu da pasif olmalı.
            self.set_game_buttons_enabled(True) # Oyunla ilgili ana butonları aktif et.
            self.set_params_enabled(True) # Parametreleri aktif et.
# =====================
# Ana Uygulama Başlatıcı
# =====================
if __name__ == "__main__":
    # PyQt5 uygulamasını başlatır.
    app = QApplication(sys.argv)
    window = DroneDeliverySimulator() # Ana pencereyi oluştur.
    window.show() # Pencereyi göster.
    sys.exit(app.exec_()) # Uygulama döngüsünü başlat ve çıkışta temizle.