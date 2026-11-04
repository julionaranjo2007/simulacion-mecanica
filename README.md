import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# ==========================================
# 1. PARÁMETROS DEL SISTEMA
# ==========================================
M = 600.0          # Masa de la pluma (kg)
m = 80.0           # Masa del gancho (kg)
L = 6.0            # Longitud de la pluma (m)
l = 1.8            # Longitud del cable del gancho (m)
H = 12.0           # Altura inicial aumentada para apreciar mejor la oscilación (m)
g = 9.81           # Aceleración de la gravedad (m/s^2)
I = M * (L**2) / 12.0  # Momento de inercia respecto al CM

# ==========================================
# 2. CONDICIONES INICIALES (Gancho con movimiento pronunciado)
# ==========================================
xC0 = L / 2.0                 # Posición X del CM
yC0 = H                       # Altura Y del CM
theta0 = 0.0                  # Pluma horizontal
psi0 = np.radians(45.0)       # Gancho fuertemente desviado (45°)
vx0 = 0.0                     # Sin velocidad horizontal inicial
vy0 = 0.0                     # Caída desde el reposo
omega0 = -0.3                 # Giro suave de la pluma
psiDot0 = 3.0                 # Velocidad angular inicial alta para el gancho

# Vector de estado: [xC, yC, theta, psi, vx, vy, omega, psiDot]
z0 = [xC0, yC0, theta0, psi0, vx0, vy0, omega0, psiDot0]

# ==========================================
# 3. ECUACIONES DE MOVIMIENTO DE LAGRANGE
# ==========================================
def ecuaciones_caida_libre(t, z):
    xC, yC, theta, psi, vx, vy, omega, psiDot = z
    Mt = M + m
    
    # Matriz de Masa Acoplada 4x4
    MM = np.array([
        [Mt,  0.0, 0.0, m * l * np.cos(psi)],
        [0.0, Mt,  0.0, m * l * np.sin(psi)],
        [0.0, 0.0, I,   0.0],
        [m * l * np.cos(psi), m * l * np.sin(psi), 0.0, m * (l**2)]
    ])
    
    # Vector de Fuerzas Generalizadas
    rhs = np.array([
        m * l * np.sin(psi) * (psiDot**2),
        -Mt * g - m * l * np.cos(psi) * (psiDot**2),
        0.0,
        -m * g * l * np.sin(psi)
    ])
    
    # Resolver aceleraciones
    acc = np.linalg.solve(MM, rhs)
    axC, ayC, thetaDDot, psiDDot = acc[0], acc[1], acc[2], acc[3]
    
    return [vx, vy, omega, psiDot, axC, ayC, thetaDDot, psiDDot]

# Detectar impacto con el suelo
def evento_suelo(t, z):
    xC, yC, theta, psi = z[0], z[1], z[2], z[3]
    yIzq = yC - (L / 2.0) * np.sin(theta)
    yDer = yC + (L / 2.0) * np.sin(theta)
    yGancho = yC - l * np.cos(psi)
    return min(yIzq, yDer, yGancho)

evento_suelo.terminal = True
evento_suelo.direction = -1

# ==========================================
# 4. INTEGRACIÓN NUMÉRICA
# ==========================================
tFinal = 2.0
t_eval = np.linspace(0, tFinal, 600)

sol = solve_ivp(
    ecuaciones_caida_libre,
    [0, tFinal],
    z0,
    method='RK45',
    t_eval=t_eval,
    events=evento_suelo,
    rtol=1e-9,
    atol=1e-10
)

t = sol.t
xC, yC = sol.y[0], sol.y[1]
theta, psi = sol.y[2], sol.y[3]

# ==========================================
# 5. ANIMACIÓN
# ==========================================
fig, ax = plt.subplots(figsize=(9, 8))
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlim(-2, L + 4)
ax.set_ylim(-1, H + 2)
ax.set_xlabel('Posición X (m)')
ax.set_ylabel('Posición Y (m)')
ax.set_title('Dinámica Acoplada: Caída de la Pluma y Oscilación Activa del Gancho')

# Estructura fija (Torre y suelo)
ax.plot([0, 0], [0, H], color='gray', linewidth=7)
ax.plot([-2, L + 4], [0, 0], 'k-', linewidth=2)

# Elementos móviles
line_barra, = ax.plot([], [], '-', color='darkorange', linewidth=6, label='Pluma')
point_anclaje, = ax.plot([], [], 'ko', markerfacecolor='steelblue', markersize=6)
line_cable, = ax.plot([], [], 'k-', linewidth=2)
point_gancho, = ax.plot([], [], 'o', markersize=11, markerfacecolor='red', markeredgecolor='k', label='Gancho')

# Estela del gancho para ver claramente sus oscilaciones
trail_gancho, = ax.plot([], [], 'b:', linewidth=1.5, label='Trayectoria del Gancho')
ax.legend(loc='upper right')

time_text = ax.text(-1.5, H + 1.2, '', fontsize=11, fontweight='bold')

def init():
    line_barra.set_data([], [])
    point_anclaje.set_data([], [])
    line_cable.set_data([], [])
    point_gancho.set_data([], [])
    trail_gancho.set_data([], [])
    time_text.set_text('')
    return line_barra, point_anclaje, line_cable, point_gancho, trail_gancho, time_text

def update(k):
    xc_k, yc_k = xC[k], yC[k]
    th_k, ph_k = theta[k], psi[k]
    
    # Coordenadas de la pluma
    xIzq = xc_k - (L / 2.0) * np.cos(th_k)
    yIzq = yc_k - (L / 2.0) * np.sin(th_k)
    xDer = xc_k + (L / 2.0) * np.cos(th_k)
    yDer = yc_k + (L / 2.0) * np.sin(th_k)
    
    # Posición del gancho acoplado al CM
    xG = xc_k + l * np.sin(ph_k)
    yG = yc_k - l * np.cos(ph_k)
    
    line_barra.set_data([xIzq, xDer], [yIzq, yDer])
    point_anclaje.set_data([xc_k], [yc_k])
    line_cable.set_data([xc_k, xG], [yc_k, yG])
    point_gancho.set_data([xG], [yG])
    
    # Actualizar estela del gancho
    xG_hist = xC[:k+1] + l * np.sin(psi[:k+1])
    yG_hist = yC[:k+1] - l * np.cos(psi[:k+1])
    trail_gancho.set_data(xG_hist, yG_hist)
    
    time_text.set_text(f't = {t[k]:.2f} s | Ángulo Gancho (ψ) = {np.degrees(ph_k):.1f}°')
    return line_barra, point_anclaje, line_cable, point_gancho, trail_gancho, time_text

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, interval=20, blit=True)
plt.show()
