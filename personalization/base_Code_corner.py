
import pygame
import random
import math
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# func for the startposition : 60 % corners and 40 % others place
def generate_random_start_position(
    width,
    height,
    radius,
    corner_probability=0.60,
    corner_size=0.15
):

    # --------------------------------------------------
    # 60 %: Punkt aus einem der vier Eckbereiche
    # --------------------------------------------------

    if random.random() < corner_probability:

        corner = random.choices(
            [
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right"
            ],
            weights=[
                1,
                1,
                1,
                1
            ]
        )[0]                                # maybe after first Exp we need another Distribution, we can change weights

        corner_width = width * corner_size
        corner_height = height * corner_size

        if corner == "top_left":

            x = random.randint(
                radius,
                int(corner_width)
            )

            y = random.randint(
                radius,
                int(corner_height)
            )

        elif corner == "top_right":

            x = random.randint(
                int(width - corner_width),
                width - radius
            )

            y = random.randint(
                radius,
                int(corner_height)
            )

        elif corner == "bottom_left":

            x = random.randint(
                radius,
                int(corner_width)
            )

            y = random.randint(
                int(height - corner_height),
                height - radius
            )

        else:  # bottom_right

            x = random.randint(
                int(width - corner_width),
                width - radius
            )

            y = random.randint(
                int(height - corner_height),
                height - radius
            )


    # --------------------------------------------------
    # 40 %: gesamte Bildschirmfläche
    # --------------------------------------------------

    else:

        x = random.randint(
            radius,
            width - radius
        )

        y = random.randint(
            radius,
            height - radius
        )

    return x, y



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
camera_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera_Width:   {frame_width}")
print(f"Camera_Height:  {frame_height}")
print(f"Camera_FPS:     {camera_fps}")

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


FPS = 30

# Uhr für konstante framerate
clock = pygame.time.Clock()

# --------------------------------
# Punkt-Eigenschaften
# --------------------------------

radius = 8
speed = 10

RESET_INTERVAL = 2.0
SETTLE_TIME = 0.2

RESET_INTERVAL_MS = int(
    RESET_INTERVAL * 1000
)

SETTLE_TIME_MS = int(
    SETTLE_TIME * 1000
)


CORNER_PROBABILITY = 0.60
CORNER_SIZE = 0.15


# --------------------------------
# Erste Position
# --------------------------------

x, y = generate_random_start_position(
    WIDTH,
    HEIGHT,
    radius,
    corner_probability=CORNER_PROBABILITY,
    corner_size=CORNER_SIZE
)

angle = random.uniform(
    0,
    2 * math.pi
)

vx = speed * math.cos(angle)
vy = speed * math.sin(angle)


moving = False

frame_count = 0
segment_id = 0

last_random_change = pygame.time.get_ticks()

phase = "wait"


# --------------------------------
# Positionsdatei
# --------------------------------

position_file = open(
    f"{record_number}_position.txt",
    "w"
)

position_file.write(
    "segment_id,frame,x,y,phase\n"
)


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

                # Neuer Start des Experiments
                frame_count = 0
                segment_id = 0

                # Timer für ersten Punkt starten
                last_random_change = pygame.time.get_ticks()

                # Erster Punkt befindet sich zunächst in der Wartephase
                phase = "wait"

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                video_writer = cv2.VideoWriter(
                    f'{record_number}_{diagram_name}_video.mp4',
                    fourcc,
                    FPS,
                    (frame_width, frame_height),
                )

                # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    f'{record_number}_{diagram_name}_video.mp4',
                    fourcc,
                    FPS,
                    (frame_width, frame_height),
                )


            # Event (Bewegungen_stop)
            if event.key == pygame.K_q:
                running = False

    # Bewegung, nur wenn gestartet
    if moving:

        current_time = pygame.time.get_ticks()

        elapsed_time = (
                current_time - last_random_change
        )

        # ==========================================================
        # 1. Nach 0.2 Sekunden: WAIT → MOVE
        # ==========================================================

        if (
                phase == "wait"
                and elapsed_time >= SETTLE_TIME_MS
        ):
            phase = "move"

        # ==========================================================
        # 2. Nach 2 Sekunden: neuer Punkt / neues Segment
        # ==========================================================

        if elapsed_time >= RESET_INTERVAL_MS:
            # ------------------------------------------------------
            # Neue Segment-ID
            # ------------------------------------------------------

            segment_id += 1

            # ------------------------------------------------------
            # Neue zufällige Startposition
            # ------------------------------------------------------

            x, y = generate_random_start_position(
                WIDTH,
                HEIGHT,
                radius,
                corner_probability=CORNER_PROBABILITY,
                corner_size=CORNER_SIZE
            )

            # ------------------------------------------------------
            # Neue zufällige Richtung
            # ------------------------------------------------------

            angle = random.uniform(
                0,
                2 * math.pi
            )

            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)

            # ------------------------------------------------------
            # Neues Segment beginnt mit WAIT
            # ------------------------------------------------------

            phase = "wait"

            # ------------------------------------------------------
            # Timer zurücksetzen
            # ------------------------------------------------------

            last_random_change = current_time

        # ==========================================================
        # 3. Bewegung nur in der MOVE-Phase
        # ==========================================================

        if phase == "move":

            x += vx
            y += vy

            # ------------------------------------------------------
            # Wand-Kollision X
            # ------------------------------------------------------

            if x - radius <= 0:

                x = radius
                vx = abs(vx)

            elif x + radius >= WIDTH:

                x = WIDTH - radius
                vx = -abs(vx)

            # ------------------------------------------------------
            # Wand-Kollision Y
            # ------------------------------------------------------

            if y - radius <= 0:

                y = radius
                vy = abs(vy)

            elif y + radius >= HEIGHT:

                y = HEIGHT - radius
                vy = -abs(vy)

        # ==========================================================
        # 4. Jeden Frame speichern
        # ==========================================================

        position_file.write(
            f"{segment_id},"
            f"{frame_count},"
            f"{x:.6f},"
            f"{y:.6f},"
            f"{phase}\n"
        )

        frame_count += 1


    screen.fill((0, 0, 0))  # Bildschirm löschen
    pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), radius)  # Punkt zeichen
    pygame.display.flip()  # Aktualisierung

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
    segment_id, frame, x, y, phase = line.strip().split(",")

    # frames.append(int(frame))
    # x_positions.append(float(x))
    # y_positions.append(float(y))
    # Nur einmal pro Sekunde
    if int(frame) % FPS == 0:
        x_positions.append(float(x) / 10)
        y_positions.append(float(y) / 10)

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
