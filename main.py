import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# KONFIGURASI PATH & PARAMETER
# =========================================================
INPUT_FOLDER = "citra"
OUTPUT_BASE = "hasil"

GLOBAL_THRESHOLD_VAL = 127
THRESHOLD_PERCENTAGE = 2.0  # Batas minimal rasio piksel %

KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Folder Penyimpanan Hasil
FOLDERS = [
    "1_citra_asli", "2_grayscale", 
    "3_global_thresh", "4_global_opening", "5_global_closing",
    "6_otsu_thresh", "7_otsu_opening", "8_otsu_closing",
    "9_perbandingan"
]

for folder in FOLDERS:
    os.makedirs(os.path.join(OUTPUT_BASE, folder), exist_ok=True)

# =========================================================
# PROGRAM UTAMA
# =========================================================
def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Folder '{INPUT_FOLDER}' tidak ditemukan!")
        return

    file_list = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not file_list:
        print(f"Tidak ada gambar di dalam folder '{INPUT_FOLDER}'.")
        return

    summary_reports = []

    print("=" * 105)
    print("PROSES ANALISIS MORFOLOGI & DETEKSI TANDA TANGAN (DUAL PATH: GLOBAL & OTSU)")
    print("=" * 105)

    for file_name in file_list:
        path_img = os.path.join(INPUT_FOLDER, file_name)
        img_orig = cv2.imread(path_img)
        if img_orig is None:
            continue

        # 1. Grayscale
        gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

        # -----------------------------------------------------
        # JALUR 1: GLOBAL THRESHOLDING & MORFOLOGI
        # -----------------------------------------------------
        _, global_th = cv2.threshold(gray, GLOBAL_THRESHOLD_VAL, 255, cv2.THRESH_BINARY_INV)
        global_open = cv2.morphologyEx(global_th, cv2.MORPH_OPEN, KERNEL)
        global_close = cv2.morphologyEx(global_open, cv2.MORPH_CLOSE, KERNEL)

        # -----------------------------------------------------
        # JALUR 2: OTSU THRESHOLDING & MORFOLOGI
        # -----------------------------------------------------
        otsu_val, otsu_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_open = cv2.morphologyEx(otsu_th, cv2.MORPH_OPEN, KERNEL)
        otsu_close = cv2.morphologyEx(otsu_open, cv2.MORPH_CLOSE, KERNEL)

        # -----------------------------------------------------
        # PERHITUNGAN FOREGROUND
        # -----------------------------------------------------
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Hitung untuk Otsu
        fg_otsu = cv2.countNonZero(otsu_close)
        ratio_otsu = (fg_otsu / total_pixels) * 100

        # Hitung untuk Global (Sebagai Pengaman Gambar Kosong)
        fg_global = cv2.countNonZero(global_close)
        ratio_global = (fg_global / total_pixels) * 100

        # -----------------------------------------------------
        # KLASIFIKASI ATURAN DETEKSI (DUAL-STATUS VALIDATION)
        # -----------------------------------------------------
        if ratio_global < 0.5 or (otsu_val > 220 and ratio_otsu > 15.0):
            status = "SIGNATURE ABSENT"
            fg_final = fg_global
            ratio_final = ratio_global
        else:
            status = "SIGNATURE PRESENT" if ratio_otsu > THRESHOLD_PERCENTAGE else "SIGNATURE ABSENT"
            fg_final = fg_otsu
            ratio_final = ratio_otsu

        # -----------------------------------------------------
        # SIMPAN GAMBAR SATUAN KE MASING-MASING FOLDER
        # -----------------------------------------------------
        cv2.imwrite(os.path.join(OUTPUT_BASE, "1_citra_asli", file_name), img_orig)
        cv2.imwrite(os.path.join(OUTPUT_BASE, "2_grayscale", file_name), gray)
        
        cv2.imwrite(os.path.join(OUTPUT_BASE, "3_global_thresh", file_name), global_th)
        cv2.imwrite(os.path.join(OUTPUT_BASE, "4_global_opening", file_name), global_open)
        cv2.imwrite(os.path.join(OUTPUT_BASE, "5_global_closing", file_name), global_close)

        cv2.imwrite(os.path.join(OUTPUT_BASE, "6_otsu_thresh", file_name), otsu_th)
        cv2.imwrite(os.path.join(OUTPUT_BASE, "7_otsu_opening", file_name), otsu_open)
        cv2.imwrite(os.path.join(OUTPUT_BASE, "8_otsu_closing", file_name), otsu_close)

        # -----------------------------------------------------
        # VISUALISASI PERBANDINGAN LENGKAP (2 BARIS x 5 KOLOM)
        # -----------------------------------------------------
        h, w = gray.shape
        aspect_ratio = w / h

        fig, axes = plt.subplots(2, 5, figsize=(20, 20 / aspect_ratio))
        fig.suptitle(f"Analisis Deteksi Tanda Tangan: {file_name}", fontsize=13, fontweight='bold')

        # Baris 1: Citra Input & Jalur Global Thresholding
        axes[0, 0].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB), aspect='equal')
        axes[0, 0].set_title("Citra Asli", fontsize=10)
        
        axes[0, 1].imshow(gray, cmap='gray', aspect='equal')
        axes[0, 1].set_title("Grayscale", fontsize=10)

        axes[0, 2].imshow(global_th, cmap='gray', aspect='equal')
        axes[0, 2].set_title(f"Global Thresh (T={GLOBAL_THRESHOLD_VAL})", fontsize=10)

        axes[0, 3].imshow(global_open, cmap='gray', aspect='equal')
        axes[0, 3].set_title("Global Opening", fontsize=10)

        axes[0, 4].imshow(global_close, cmap='gray', aspect='equal')
        axes[0, 4].set_title("Global Closing", fontsize=10)

        # Baris 2: Jalur Otsu Thresholding + Opening + Closing + Status Keputusan
        axes[1, 0].imshow(otsu_th, cmap='gray', aspect='equal')
        axes[1, 0].set_title(f"Otsu Thresh (T={int(otsu_val)})", fontsize=10)

        axes[1, 1].imshow(otsu_open, cmap='gray', aspect='equal')
        axes[1, 1].set_title("Otsu Opening", fontsize=10)

        axes[1, 2].imshow(otsu_close, cmap='gray', aspect='equal')
        axes[1, 2].set_title("Otsu Closing", fontsize=10)

        # Hilangkan panel kosong di 1,3
        axes[1, 3].axis('off')

        # Panel Status Keputusan di 1,4
        axes[1, 4].axis('off')
        color = 'green' if status == "SIGNATURE PRESENT" else 'red'
        axes[1, 4].text(0.5, 0.65, status, color=color, fontsize=12, fontweight='bold', 
                        ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", ec=color, fc="none", lw=2))
        axes[1, 4].text(0.5, 0.40, f"Piksel: {fg_final} px", fontsize=10, ha='center', va='center')
        axes[1, 4].text(0.5, 0.25, f"Rasio: {ratio_final:.2f}%", fontsize=10, ha='center', va='center')

        # Hilangkan sumbu koordinat untuk semua gambar
        for ax_row in axes:
            for ax in ax_row:
                if ax not in [axes[1, 3], axes[1, 4]]:
                    ax.axis('off')

        plt.tight_layout()
        plot_name = f"Perbandingan_{os.path.splitext(file_name)[0]}.png"
        plt.savefig(os.path.join(OUTPUT_BASE, "9_perbandingan", plot_name), dpi=150, bbox_inches='tight')
        plt.close()

        # Simpan rekap untuk tabel terminal
        summary_reports.append({
            'filename': file_name,
            'global_t': GLOBAL_THRESHOLD_VAL,
            'otsu_t': int(otsu_val),
            'fg_px': fg_final,
            'pct': ratio_final,
            'status': status
        })

        print(f"Diproses: {file_name:<40} -> Status: {status}")

    # =========================================================
    # REKAPITULASI TABEL LENGKAP DI TERMINAL
    # =========================================================
    print("\n" + "=" * 105)
    print("                               REKAPITULASI HASIL SEGMENTASI & DETEKSI")
    print("=" * 105)
    print(f"{'No':<4} | {'Nama Gambar':<35} | {'Global T':<9} | {'Otsu T':<8} | {'Piksel FG':<10} | {'Rasio (%)':<9} | {'Status':<18}")
    print("-" * 105)

    for idx, r in enumerate(summary_reports, 1):
        print(f"{idx:<4} | {r['filename'][:34]:<35} | {r['global_t']:<9} | {r['otsu_t']:<8} | {r['fg_px']:<10} | {r['pct']:<9.2f} | {r['status']:<18}")

    print("=" * 105)
    print(f"\n[SELESAI] Hasil analisis perbandingan lengkap telah disimpan di '{OUTPUT_BASE}/9_perbandingan/'.")

if __name__ == "__main__":
    main()