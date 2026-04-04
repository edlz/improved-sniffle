"""
debug_ram.py — manually play the game and observe RAM values

Controls:
    Arrow keys  — cursor movement
    Z           — A button (confirm)
    X           — B button (cancel)
    A           — X button
    S           — Y button
    D           — L button
    F           — R button
    Return      — START
    Tab         — SELECT
    F5          — save state
    F9          — load state
    F6          — start/stop recording demo
    Q           — quit
"""

from pathlib import Path
import time
import pygame
import numpy as np
import stable_retro
from envs.wrappers import RewardWrapper, RAMObsWrapper

stable_retro.data.add_custom_integration(str(Path("retro_data").resolve()))

GAME = "FE776-Snes"
STATE = "debug_save"
SCALE = 3
DEMO_DIR = Path("demos")
DEMO_DIR.mkdir(exist_ok=True)

raw_env = stable_retro.make(
    game=GAME,
    state=STATE,
    use_restricted_actions=stable_retro.Actions.ALL,
    render_mode="rgb_array",
    inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
)
env = RewardWrapper(raw_env)
env.reset()
_, _, _, _, info = env.step(np.zeros(12, dtype=np.int8))

# Use RAMObsWrapper's extract logic to build obs from info
_ram_extractor = RAMObsWrapper.__new__(RAMObsWrapper)

def extract_obs(info):
    return _ram_extractor._extract(info)

# Discrete action mapping: keyboard -> discrete action index
# Matches FEThracia776DiscretizerSmall combos
DISCRETE_MAP = {
    frozenset():          0,  # NOOP
    frozenset(["UP"]):    1,
    frozenset(["DOWN"]):  2,
    frozenset(["LEFT"]):  3,
    frozenset(["RIGHT"]): 4,
    frozenset(["A"]):     5,
    frozenset(["B"]):     6,
    frozenset(["R"]):     7,
    frozenset(["START"]): 8,
}

buttons = env.unwrapped.buttons

KEY_MAP = {
    pygame.K_UP: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_RIGHT: "RIGHT",
    pygame.K_z: "A",
    pygame.K_x: "B",
    pygame.K_f: "R",
    pygame.K_RETURN: "START",
}

SAVE_PATH = "checkpoints/debug_save.state"

pygame.init()
img = env.unwrapped.get_screen()
h, w = img.shape[:2]
INFO_HEIGHT = 260
screen = pygame.display.set_mode((w * SCALE, h * SCALE + INFO_HEIGHT))
pygame.display.set_caption("Thracia 776 — RAM Debug")
font = pygame.font.SysFont("monospace", 14)
clock = pygame.time.Clock()

prev_info = dict(info)
change_log = []
frame_num = 0
running = True
recording = False
demo_obs = []
demo_actions = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_F5:
                import gzip
                with open(SAVE_PATH, "wb") as f:
                    f.write(gzip.compress(env.unwrapped.em.get_state()))
                change_log.append(f"[{frame_num:>6}] *** SAVED to {SAVE_PATH} ***")
            elif event.key == pygame.K_F6:
                recording = not recording
                if recording:
                    demo_obs = []
                    demo_actions = []
                    change_log.append(f"[{frame_num:>6}] *** RECORDING STARTED ***")
                else:
                    if demo_obs:
                        fname = DEMO_DIR / f"demo_{int(time.time())}.npz"
                        np.savez(fname,
                            obs=np.array(demo_obs),
                            actions=np.array(demo_actions))
                        change_log.append(f"[{frame_num:>6}] *** SAVED {len(demo_obs)} frames to {fname} ***")
                    else:
                        change_log.append(f"[{frame_num:>6}] *** RECORDING EMPTY — not saved ***")
            elif event.key == pygame.K_F9:
                import gzip
                try:
                    with open(SAVE_PATH, "rb") as f:
                        env.unwrapped.em.set_state(gzip.decompress(f.read()))
                    change_log.append(f"[{frame_num:>6}] *** LOADED from {SAVE_PATH} ***")
                except FileNotFoundError:
                    change_log.append(f"[{frame_num:>6}] *** No save found ***")

    keys = pygame.key.get_pressed()
    action = np.zeros(12, dtype=np.int8)
    pressed = []
    for key, btn in KEY_MAP.items():
        if keys[key]:
            action[buttons.index(btn)] = 1
            pressed.append(btn)

    prev_info = dict(info)
    obs, reward, terminated, truncated, info = env.step(action)
    frame_num += 1

    # Record demo data
    if recording:
        pressed_set = frozenset(pressed)
        discrete_action = DISCRETE_MAP.get(pressed_set, 0)
        demo_obs.append(extract_obs(info))
        demo_actions.append(discrete_action)

    if terminated or truncated:
        env.reset()
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.int8))

    # Track changes
    changes = []
    watch_keys = ["selected_char", "phase", "cursor_x", "cursor_y", "turn"]
    for i in range(5):
        watch_keys += [f"p{i}_x", f"p{i}_y", f"p{i}_hp"]
    for i in range(15):
        watch_keys += [f"e{i}_x", f"e{i}_y", f"e{i}_hp"]
    for k in watch_keys:
        old = prev_info.get(k)
        new = info.get(k)
        if old != new:
            changes.append(f"{k}:{old}->{new}")

    if changes or reward != 0:
        entry = f"[{frame_num:>6}] "
        if pressed:
            entry += f"{','.join(pressed):>8} "
        else:
            entry += "         "
        if reward != 0:
            entry += f"r={reward:+.3f} "
        entry += "  ".join(changes)
        change_log.append(entry)
        if len(change_log) > 50:
            change_log.pop(0)

    # Draw game frame
    frame = env.unwrapped.get_screen()
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (w * SCALE, h * SCALE))
    screen.fill((20, 20, 20))
    screen.blit(surf, (0, 0))

    # Draw current state
    y = h * SCALE + 4
    state_line = (
        f"cursor=({info.get('cursor_x','?')},{info.get('cursor_y','?')})  "
        f"turn={info.get('turn','?')}  phase={info.get('phase','?')}  "
        f"sel_char={info.get('selected_char','?')}  "
        f"pressed={','.join(pressed) or 'none'}"
        f"{'  *** REC ***' if recording else ''}"
    )
    screen.blit(font.render(state_line, True, (200, 200, 200)), (8, y))

    unit_line = (
        f"p0=({info.get('p0_x')},{info.get('p0_y')}) hp={info.get('p0_hp')}/{info.get('p0_maxhp')}  "
        f"p1=({info.get('p1_x')},{info.get('p1_y')}) hp={info.get('p1_hp')}/{info.get('p1_maxhp')}  "
        f"p2=({info.get('p2_x')},{info.get('p2_y')}) hp={info.get('p2_hp')}/{info.get('p2_maxhp')}"
    )
    screen.blit(font.render(unit_line, True, (200, 200, 200)), (8, y + 16))

    # Draw scrolling change log
    screen.blit(font.render("--- change log ---", True, (150, 150, 150)), (8, y + 36))
    visible = change_log[-12:]
    for i, entry in enumerate(visible):
        color = (100, 255, 100) if "r=" in entry else (180, 180, 180)
        screen.blit(font.render(entry, True, color), (8, y + 52 + i * 16))

    pygame.display.flip()
    clock.tick(60)

env.close()
pygame.quit()
