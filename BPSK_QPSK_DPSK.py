
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# NORMALIZED BPSK, GRAY-CODED QPSK AND DBPSK
# First-principles implementation using NumPy and Matplotlib
# ============================================================

np.random.seed(42)

N = 100_000
EbN0_dB = 8
phase_offsets = [0, 10, 30, 60, 90]

# ------------------------------------------------------------
# 1. COMMON FUNCTIONS
# ------------------------------------------------------------

def add_awgn(symbols, ebn0_db, bits_per_symbol=1):
    """Add complex AWGN to unit-energy symbols."""
    ebn0 = 10 ** (ebn0_db / 10)
    Es = np.mean(np.abs(symbols) ** 2)
    N0 = Es / (bits_per_symbol * ebn0)
    noise = np.sqrt(N0 / 2) * (
        np.random.randn(len(symbols))
        + 1j * np.random.randn(len(symbols))
    )
    return symbols + noise


def ber(tx_bits, rx_bits):
    """Bit error rate."""
    return np.mean(tx_bits != rx_bits)


# ------------------------------------------------------------
# 2. BPSK MAPPER AND COHERENT DEMAPPER
# ------------------------------------------------------------

def bpsk_map(bits):
    # Bit 0 -> +1; bit 1 -> -1. Average symbol energy = 1.
    return 1 - 2 * bits


def bpsk_demapper(received, phase_estimate=0):
    # Compensate estimated carrier phase, then decide by sign.
    corrected = received * np.exp(-1j * np.deg2rad(phase_estimate))
    return (corrected.real < 0).astype(int)


# ------------------------------------------------------------
# 3. GRAY-CODED QPSK MAPPER AND DEMAPPER
# ------------------------------------------------------------

def qpsk_map(bits):
    # Gray mapping:
    # 00 -> (+1 + j)/sqrt(2)
    # 01 -> (+1 - j)/sqrt(2)
    # 10 -> (-1 + j)/sqrt(2)
    # 11 -> (-1 - j)/sqrt(2)
    pairs = bits.reshape(-1, 2)
    i = 1 - 2 * pairs[:, 0]
    q = 1 - 2 * pairs[:, 1]
    return (i + 1j * q) / np.sqrt(2)


def qpsk_demapper(received, phase_estimate=0):
    corrected = received * np.exp(-1j * np.deg2rad(phase_estimate))
    bits_i = (corrected.real < 0).astype(int)
    bits_q = (corrected.imag < 0).astype(int)
    return np.column_stack((bits_i, bits_q)).ravel()


# ------------------------------------------------------------
# 4. DBPSK DIFFERENTIAL ENCODER AND DETECTOR
# ------------------------------------------------------------

def dbpsk_encode(bits):
    # Bit 0: retain phase; bit 1: reverse phase by 180 degrees.
    # Reference symbol is +1.
    symbols = np.empty(len(bits) + 1, dtype=complex)
    symbols[0] = 1 + 0j

    for k, bit in enumerate(bits):
        symbols[k + 1] = symbols[k] * (1 - 2 * bit)

    return symbols


def dbpsk_detect(received):
    # Compare adjacent symbols: r[k] * conjugate(r[k-1]).
    # A negative real part indicates a phase reversal (bit 1).
    statistic = received[1:] * np.conj(received[:-1])
    detected_bits = (statistic.real < 0).astype(int)
    return detected_bits, statistic


# ------------------------------------------------------------
# 5. GENERATE DATA AND MODULATED SIGNALS
# ------------------------------------------------------------

bits_bpsk = np.random.randint(0, 2, N)
bits_qpsk = np.random.randint(0, 2, 2 * N)
bits_dbpsk = np.random.randint(0, 2, N)

symbols_bpsk = bpsk_map(bits_bpsk)
symbols_qpsk = qpsk_map(bits_qpsk)
symbols_dbpsk = dbpsk_encode(bits_dbpsk)

# Normalize all modulation constellations to unit symbol energy.
print("Average symbol energies:")
print("BPSK :", np.mean(np.abs(symbols_bpsk) ** 2))
print("QPSK :", np.mean(np.abs(symbols_qpsk) ** 2))
print("DBPSK:", np.mean(np.abs(symbols_dbpsk[1:]) ** 2))

# ------------------------------------------------------------
# 6. BER UNDER CARRIER PHASE OFFSETS
# ------------------------------------------------------------

ber_bpsk = []
ber_qpsk = []
ber_dbpsk = []

for offset in phase_offsets:
    theta = np.deg2rad(offset)

    # Apply carrier phase offset and independent AWGN.
    r_bpsk = add_awgn(
        symbols_bpsk * np.exp(1j * theta), EbN0_dB
    )
    r_qpsk = add_awgn(
        symbols_qpsk * np.exp(1j * theta), EbN0_dB, 2
    )
    r_dbpsk = add_awgn(
        symbols_dbpsk * np.exp(1j * theta), EbN0_dB
    )

    # Coherent detection assumes the receiver knows phase = 0.
    # Therefore the imposed offset remains uncompensated.
    b_hat = bpsk_demapper(r_bpsk, phase_estimate=0)
    q_hat = qpsk_demapper(r_qpsk, phase_estimate=0)
    d_hat, d_stat = dbpsk_detect(r_dbpsk)

    ber_bpsk.append(ber(bits_bpsk, b_hat))
    ber_qpsk.append(ber(bits_qpsk, q_hat))
    ber_dbpsk.append(ber(bits_dbpsk, d_hat))

print(f"\nBER at Eb/N0 = {EbN0_dB} dB")
print("Phase | BPSK BER | QPSK BER | DBPSK BER")
print("-----------------------------------------")
for i, offset in enumerate(phase_offsets):
    print(
        f"{offset:>3}°  | "
        f"{ber_bpsk[i]:.6f} | "
        f"{ber_qpsk[i]:.6f} | "
        f"{ber_dbpsk[i]:.6f}"
    )

# ------------------------------------------------------------
# 7. VISUALIZATION 1: IDEAL AND NOISY CONSTELLATIONS
# ------------------------------------------------------------

r_bpsk_plot = add_awgn(symbols_bpsk[:300] * np.exp(1j*np.deg2rad(30)),
                       EbN0_dB)
r_qpsk_plot = add_awgn(symbols_qpsk[:300] * np.exp(1j*np.deg2rad(30)),
                       EbN0_dB, 2)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].scatter(symbols_bpsk[:300].real,
                   symbols_bpsk[:300].imag, s=12)
axes[0, 0].set_title("Ideal BPSK Constellation")
axes[0, 1].scatter(r_bpsk_plot.real, r_bpsk_plot.imag, s=12)
axes[0, 1].set_title("Noisy BPSK (30° offset)")

axes[1, 0].scatter(symbols_qpsk[:300].real,
                   symbols_qpsk[:300].imag, s=12)
axes[1, 0].set_title("Ideal Gray-Coded QPSK")
axes[1, 1].scatter(r_qpsk_plot.real, r_qpsk_plot.imag, s=12)
axes[1, 1].set_title("Noisy QPSK (30° offset)")

for ax in axes.ravel():
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.axvline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="datalim")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. VISUALIZATION 2: QPSK MAPPING
# ------------------------------------------------------------

mapping_bits = ["00", "01", "10", "11"]
mapping_symbols = qpsk_map(np.array([0, 0, 0, 1, 1, 0, 1, 1]))

plt.figure(figsize=(7, 6))
plt.scatter(mapping_symbols.real, mapping_symbols.imag, s=100)

for label, symbol in zip(mapping_bits, mapping_symbols):
    plt.annotate(label, (symbol.real, symbol.imag),
                 xytext=(8, 8), textcoords="offset points")

plt.axhline(0, color="gray")
plt.axvline(0, color="gray")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("Gray-Coded QPSK Mapping")
plt.grid(True)
plt.axis("equal")
plt.show()

# ------------------------------------------------------------
# 9. VISUALIZATION 3: DBPSK PHASE TRAJECTORY
# ------------------------------------------------------------

num_symbols = 50
phase = np.angle(symbols_dbpsk[:num_symbols])
phase_unwrapped = np.unwrap(phase)

plt.figure(figsize=(10, 4))
plt.step(np.arange(num_symbols), np.rad2deg(phase_unwrapped),
         where="post")
plt.xlabel("Symbol Index")
plt.ylabel("Unwrapped Phase (degrees)")
plt.title("DBPSK Transmitted Phase Trajectory")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 10. VISUALIZATION 4: BER COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))
plt.semilogy(phase_offsets, ber_bpsk, "o-", label="Coherent BPSK")
plt.semilogy(phase_offsets, ber_qpsk, "s-", label="Coherent Gray QPSK")
plt.semilogy(phase_offsets, ber_dbpsk, "^-", label="Differential DBPSK")
plt.xlabel("Carrier Phase Offset (degrees)")
plt.ylabel("Bit Error Rate (BER)")
plt.title(f"BER vs Carrier Phase Offset (Eb/N0 = {EbN0_dB} dB)")
plt.xticks(phase_offsets)
plt.ylim(1e-5, 1)
plt.grid(True, which="both")
plt.legend()
plt.show()

# ------------------------------------------------------------
# 11. VISUALIZATION 5: DECISION STATISTICS
# ------------------------------------------------------------

# BPSK coherent decision statistic at 30 degrees.
theta = np.deg2rad(30)
r = add_awgn(symbols_bpsk * np.exp(1j * theta), EbN0_dB)
coherent_stat = r.real

# DBPSK differential decision statistic at 30 degrees.
rd = add_awgn(symbols_dbpsk * np.exp(1j * theta), EbN0_dB)
_, differential_stat = dbpsk_detect(rd)

plt.figure(figsize=(10, 5))
plt.hist(coherent_stat[bits_bpsk == 0], bins=60, density=True,
         alpha=0.55, label="BPSK bit 0")
plt.hist(coherent_stat[bits_bpsk == 1], bins=60, density=True,
         alpha=0.55, label="BPSK bit 1")
plt.xlabel("Coherent Decision Statistic: Re{r}")
plt.ylabel("Probability Density")
plt.title("BPSK Decision Statistics (30° Offset)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(differential_stat.real[bits_dbpsk == 0], bins=60,
         density=True, alpha=0.55, label="DBPSK bit 0")
plt.hist(differential_stat.real[bits_dbpsk == 1], bins=60,
         density=True, alpha=0.55, label="DBPSK bit 1")
plt.xlabel("Differential Statistic: Re{r[k]r*[k-1]}")
plt.ylabel("Probability Density")
plt.title("DBPSK Decision Statistics (30° Offset)")
plt.legend()
plt.grid(True)
plt.show()

print("\nSimulation completed successfully.")
