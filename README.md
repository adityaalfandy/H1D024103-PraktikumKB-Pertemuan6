# Neural Network dari Scratch — Perceptron & Backpropagation

Proyek ini berisi implementasi dua algoritma jaringan saraf dasar menggunakan Python dan NumPy: **Perceptron** dan **Backpropagation**. Masing-masing algoritma terdiri dari dua file: file kelas utama dan file eksekusi/pengujian.

---

## Daftar File

| File | Deskripsi |
|---|---|
| `Perceptron.py` | Implementasi kelas Perceptron |
| `Perceptron_or.py` | Eksekusi Perceptron untuk masalah logika OR |
| `Backpropagation.py` | Implementasi kelas Backpropagation |
| `Backpropagation_xor.py` | Eksekusi Backpropagation untuk masalah logika XOR |

---

## 1. `Perceptron.py` — Kelas Perceptron

### Gambaran Umum
File ini mendefinisikan kelas `Perceptron`, sebuah jaringan saraf satu lapis yang mampu menyelesaikan masalah yang **linearly separable** (dapat dipisahkan secara linear), seperti gerbang logika AND dan OR.

### Atribut
| Atribut | Deskripsi |
|---|---|
| `alpha` | Learning rate — seberapa besar langkah pembaruan bobot setiap iterasi |
| `epoch` | Jumlah maksimum iterasi pelatihan |
| `w_` | Array bobot dan bias yang digabungkan (`w_[0]` = bias, `w_[1:]` = bobot) |

### Metode

#### `__init__(self, alpha=0.1, epoch=10)`
Konstruktor yang menyimpan learning rate dan jumlah epoch. Bobot belum diinisialisasi di sini karena belum ada data masukan.

#### `weighted_sum(self, X)`
Menghitung **net input** (nilai sebelum aktivasi):
```
net = X · w + b
```
Menggunakan perkalian titik (`np.dot`) antara input dan bobot, lalu ditambah bias.

#### `predict(self, X)`
Menerapkan **fungsi aktivasi bipolar**:
- Jika `net >= 0` → output = `1`
- Jika `net < 0` → output = `-1`

#### `plot_decision_boundary(self, X, t, epoch)`
Menampilkan **garis pemisah (decision boundary)** pada setiap epoch menggunakan Matplotlib. Membantu memvisualisasikan bagaimana model belajar memisahkan dua kelas data secara bertahap.

#### `fit(self, X, t)`
Fungsi utama pelatihan Perceptron, dengan langkah-langkah:
1. Inisialisasi semua bobot dan bias = `0`
2. Untuk setiap epoch, iterasi seluruh data latih
3. Hitung prediksi (`predict`)
4. Hitung error: `error = target - prediksi`
5. Perbarui bobot dengan **Delta Rule**:
   ```
   w_baru = w_lama + alpha × error × x
   b_baru = b_lama + alpha × error
   ```
6. Simpan log pelatihan ke file `Hasil Perceptron.txt`
7. Berhenti jika **SSE (Sum Square Error) = 0** atau epoch maksimum tercapai

---

## 2. `Perceptron_or.py` — Pengujian Perceptron pada Masalah OR

### Gambaran Umum
File eksekusi yang menguji kelas `Perceptron` menggunakan dataset gerbang logika **OR Bipolar**.

### Dataset
| X1 | X2 | Target |
|---|---|---|
| 1 | 1 | 1 |
| 1 | -1 | 1 |
| -1 | 1 | 1 |
| -1 | -1 | -1 |

### Konfigurasi
```python
alpha = 0.1   # Learning rate
epoch = 10    # Maksimum epoch
```

### Cara Kerja
1. Mendefinisikan matriks input `X` dan target `t` menggunakan NumPy
2. Membuat instance kelas `Perceptron`
3. Memanggil `model.fit(X, t)` untuk memulai pelatihan
4. Hasil dan log disimpan ke `Hasil Perceptron.txt`

> **Catatan:** OR adalah masalah *linearly separable*, sehingga Perceptron dapat menyelesaikannya dengan sempurna.

---

## 3. `Backpropagation.py` — Kelas Backpropagation

### Gambaran Umum
File ini mendefinisikan kelas `Backpropagation`, sebuah jaringan saraf **multilayer** (terdiri dari input layer, hidden layer, dan output layer) yang mampu menyelesaikan masalah **non-linearly separable** seperti XOR. Menggunakan algoritma *gradient descent* dengan propagasi error ke belakang.

### Arsitektur Jaringan
```
Input Layer  →  Hidden Layer  →  Output Layer
  2 neuron       2 neuron          1 neuron
```

### Atribut
| Atribut | Deskripsi |
|---|---|
| `alpha` | Learning rate |
| `epoch` | Jumlah maksimum epoch |
| `target_error` | Ambang batas SSE agar pelatihan berhenti |
| `w_hidden` | Bobot dari input ke hidden layer (diinisialisasi random) |
| `b_hidden` | Bias hidden layer (diinisialisasi random) |
| `w_output` | Bobot dari hidden ke output layer (diinisialisasi random) |
| `b_output` | Bias output layer (diinisialisasi random) |

### Metode

#### `__init__(self, alpha, epoch, target_error)`
Konstruktor yang menyimpan hyperparameter dan menginisialisasi semua bobot serta bias secara **acak (random)** menggunakan `np.random.rand`.

#### `bi_sigmoid(self, x)`
Menerapkan **fungsi aktivasi Sigmoid Bipolar** (identik dengan `tanh`):
```
f(x) = tanh(x)
```
Output berada dalam rentang `(-1, 1)`.

#### `deriv_bi_sigmoid(self, x)`
Menghitung **turunan fungsi tanh** (dengan asumsi `x` adalah output dari `tanh`):
```
f'(x) = 1 - x²
```
Digunakan dalam proses backward propagation untuk menghitung delta/gradient.

#### `plot_error(self, x, epoch)`
Menampilkan grafik **perkembangan error (SSE)** setiap epoch menggunakan Matplotlib, dilengkapi anotasi nilai error akhir.

#### `fit(self, X, t)`
Fungsi utama pelatihan Backpropagation, terdiri dari dua fase besar:

**Fase 1 — Forward Propagation:**
1. Hitung net input hidden layer: `h_in = X · w_hidden + b_hidden`
2. Aktivasi hidden layer: `h = tanh(h_in)`
3. Hitung net input output layer: `y_in = h · w_output + b_output`
4. Aktivasi output layer: `y = tanh(y_in)`

**Fase 2 — Backward Propagation:**
1. Hitung error output: `error = target - y`
2. Hitung delta output: `d_y = error × tanh'(y)`
3. Propagasi error ke hidden layer: `error_h = d_y · w_output^T`
4. Hitung delta hidden: `d_h = error_h × tanh'(h)`
5. Perbarui bobot dan bias output layer:
   ```
   w_output += h^T · d_y × alpha
   b_output += Σ(d_y) × alpha
   ```
6. Perbarui bobot dan bias hidden layer:
   ```
   w_hidden += x^T · d_h × alpha
   b_hidden += Σ(d_h) × alpha
   ```

Hasil setiap epoch disimpan ke `hasilBackpropagation.txt`. Pelatihan berhenti jika **SSE < target_error** atau epoch maksimum tercapai.

---

## 4. `Backpropagation_xor.py` — Pengujian Backpropagation pada Masalah XOR

### Gambaran Umum
File eksekusi yang menguji kelas `Backpropagation` menggunakan dataset gerbang logika **XOR Bipolar**.

### Dataset
| X1 | X2 | Target |
|---|---|---|
| 1 | 1 | -1 |
| 1 | -1 | 1 |
| -1 | 1 | 1 |
| -1 | -1 | -1 |

### Konfigurasi
```python
alpha        = 0.3    # Learning rate
epoch        = 1000   # Maksimum epoch
target_error = 0.001  # Ambang batas SSE
```

### Cara Kerja
1. Mendefinisikan matriks input `X` dan target `t`
2. Membuat instance kelas `Backpropagation`
3. Memanggil `model.fit(X, t)` untuk memulai pelatihan
4. Hasil dan log disimpan ke `hasilBackpropagation.txt`

> **Catatan:** XOR adalah masalah *non-linearly separable* — tidak bisa diselesaikan oleh Perceptron satu lapis, namun dapat diselesaikan oleh jaringan multilayer dengan Backpropagation.

---

## Perbandingan Algoritma

| Aspek | Perceptron | Backpropagation |
|---|---|---|
| Jumlah layer | 1 (single layer) | 3 (input, hidden, output) |
| Fungsi aktivasi | Bipolar step function | Sigmoid bipolar (tanh) |
| Masalah yang bisa diselesaikan | Linearly separable | Non-linearly separable |
| Pembaruan bobot | Delta Rule sederhana | Gradient descent (chain rule) |
| Contoh penggunaan | Gerbang OR, AND | Gerbang XOR |
| Kondisi berhenti | SSE = 0 atau max epoch | SSE < target atau max epoch |

---

## Dependensi

```bash
pip install numpy matplotlib
```

---

## Cara Menjalankan

```bash
# Menjalankan Perceptron untuk masalah OR
python Perceptron_or.py

# Menjalankan Backpropagation untuk masalah XOR
python Backpropagation_xor.py
```

Output berupa:
- **File teks** (`Hasil Perceptron.txt` / `hasilBackpropagation.txt`) berisi log detail setiap epoch
- **Grafik** visualisasi decision boundary (Perceptron) atau perkembangan error (Backpropagation)