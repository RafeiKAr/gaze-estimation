
import pygame
import random
import math
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


record_number = str(input("Please enter a number for the record: "))
diagram_name = str(input("Please enter a name for the diagram: "))

# -----------------------
# Kamera initialisierung
#------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Kamera konnte nicht geöffnet werden.')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Width: {frame_width}, \t Height: {frame_height}\n")

video_writer = None
recording = False


# --------------------------------
# Pygame initialisieren
# --------------------------------
pygame.init()

# warnings.filterwarnings("ignore", category=UserWarning)

# Bildschirmgröße
info = pygame.display.Info()
# print(f"Info über Bildschirm: {info}")

# WIDTH, HEIGHT = info.current_w, info.current_h
WIDTH = info.current_w - 50
HEIGHT = info.current_h - 80
# WIDTH, HEIGHT = 800, 600
print(f"Width_screen: {WIDTH}, \t Height_screen: {HEIGHT}\n")
# 1430 and 880 (1440 , 960)
# Ubuntu (1920, 1080) --> (1870 , 1000)


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(f"Random-Startpoint with Recording: {record_number}")

# Uhr für konstante framerate
clock = pygame.time.Clock()
FPS = 60

# --------------------------------
# Punkt-Eigenschaften
# --------------------------------
radius = 8

# Zufällige Startposition
x = random.randint(radius, WIDTH - radius)
y = random.randint(radius, HEIGHT - radius)

# festen Startposition
# x = WIDTH // 2
# y = HEIGHT // 2

# Zufällige Startgeschwindigkeit
speed = 10
angle = random.uniform(0, 2 * math.pi)
vx = speed * math.cos(angle)
vy = speed * math.sin(angle)

# Bewegungszustand
moving = False
frame_count = 0

# Alle 5 Sekunden neuen Startpunkt + neue Richtung
last_random_change = pygame.time.get_ticks()
RANDOM_INTERVAL = 5000  # 5000 ms = 5 Sekunden

# ----------------------------
# Datei für Positionsdaten
# ----------------------------
position_file = open(f"{record_number}_position.txt", "w")
position_file.write("frame, x, y\n")



# --------------------------------
# Hauptschleife
# --------------------------------
running = True
while running:
    clock.tick(FPS)

    # Event (Fenster schließen)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            # Event (Bewegung_start)
            if event.key == pygame.K_s and not recording:
                moving = True
                recording = True

                last_random_change = pygame.time.get_ticks()

                # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    f'{record_number}_{diagram_name}_video.mp4',
                    fourcc,
                    30,
                    (frame_width, frame_height),
                )


            # Event (Bewegungen_stop)
            if event.key == pygame.K_q:
                running = False

    # Bewegung, nur wenn gestartet
    if moving:

        current_time = pygame.time.get_ticks()

        # Alle 5 Sekunden:
        if current_time - last_random_change >= RANDOM_INTERVAL:
            # Neuer zufälliger Startpunkt
            x = random.randint(radius, WIDTH - radius)
            y = random.randint(radius, HEIGHT - radius)

            # Neue zufällige Richtung
            angle = random.uniform(0, 2 * math.pi)

            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)

            # Timer zurücksetzen
            last_random_change = current_time

        # Punkt bewegen
        x += vx
        y += vy

        # Kollision mit Wänden
        if x - radius <= 0 or x + radius >= WIDTH:
            vx = -vx

        if y - radius <= 0 or y + radius >= HEIGHT:
            vy = -vy

        # Position pro Frame speichern
        position_file.write(
            f"{frame_count}, {x:.2f}, {y:.2f}\n"
        )

        frame_count += 1



    screen.fill((0, 0, 0))                # Bildschirm löschen
    pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), radius)       # Punkt zeichen
    pygame.display.flip()                       # Aktualisierung

    # ---------------------------
    # Kamera-Frame lesen
    # ---------------------------
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Kamera, ESC to Stop", frame)

        if recording and video_writer is not None:
            video_writer.write(frame)

    # ESC schließt Kamera-Fenster
    if cv2.waitKey(1) & 0xFF == 27:
        running = False

# -------------------------------
# Cleanup
# -------------------------------
position_file.close()

# Datei einlesen
frames = []
x_positions = []
y_positions = []

with open(f"{record_number}_position.txt", "r") as file:
    lines = file.readlines()

    # try:
    #     next(file)
    # except StopIteration:
    #     raise RuntimeError("Datei ist leer oder enthält keinen Header")
if len(lines) <= 1:
    raise RuntimeError("Keine Positionsdaten in der Datei.")

    # for line in file:
    #     frame, x, y = line.strip().split(",")
# Daten einlesen (Header überspringen)
for i, line in enumerate(lines[1:]):
    frame, x, y = line.strip().split(",")

    # frames.append(int(frame))
    # x_positions.append(float(x))
    # y_positions.append(float(y))
    # Nur einmal pro Sekunde
    if int(frame) % FPS == 0:
        x_positions.append(float(x)/10)
        y_positions.append(float(y)/10)

# Plot erstellen
plt.figure(figsize=(6, 6))

# plt.plot(x_positions, y_positions)
# time = [f / 60 for f in frames]
# plt.plot(time, x_positions, label="X(t)")
# plt.plot(time, y_positions, label="Y(t)")

# plt.plot(frames, x_positions, label="X-Position")
# plt.plot(frames, y_positions, label="Y-Position")
plt.plot(x_positions, y_positions, marker="o")
plt.xlabel("X-Position (1 Punkt pro Sekunde)")
plt.ylabel("Y-Position (1 Punkt pro Sekunde)")
plt.title("Trajektorie des Punktes (sekündlich)")

plt.legend()
plt.axis("equal")
plt.grid(True)
plt.savefig(f"{record_number}_{diagram_name}_trajectory.png")
plt.close()

if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()
pygame.quit()
