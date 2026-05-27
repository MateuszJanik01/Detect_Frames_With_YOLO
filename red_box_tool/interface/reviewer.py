"""
reviewer.py
-----------
Moduł odpowiedzialny za interfejs graficzny (GUI) narzędzia do przeglądu ramek.
Rozszerzony o wyświetlanie podpowiedzi YOLO (żółte ramki z konfidencją).
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import pyperclip
    _CLIPBOARD_OK = True
except ImportError:
    _CLIPBOARD_OK = False

from ..core.geometry import iou
from ..core.models import Box

# Parametry okna i nawigacji
WINDOW = "OCR Review"
PANEL_W = 400
ZOOM_STEP = 0.15
ZOOM_MIN = 0.1
ZOOM_MAX = 8.0

# Definicje kolorów (format BGR)
COL_OK = (30, 210, 30)
COL_REJECT = (90, 90, 90)
COL_DRAW = (0, 180, 255)
COL_MANUAL = (255, 130, 0)
COL_SELECTED = (200, 255, 255)
COL_STATUS = (20, 20, 20)
COL_PANEL = (40, 40, 45)
COL_TEXT = (220, 220, 220)
COL_HIGHLIGHT = (0, 255, 255)
COL_YOLO = (0, 220, 255)        # Żółto-cyjanowy: podpowiedzi YOLO (niezatwierdzone)
COL_YOLO_CONFIRMED = (30, 210, 30)  # Zielony po zatwierdzeniu przez użytkownika

# Kody specjalne klawiszy
KEY_ESC = 27
KEY_ENTER = (13, 10)
KEY_BACKSPACE = (8, 127)
KEY_DELETE_WIN = 3014656
KEY_DELETE_LINUX = 255
KEY_DELETE_CODES = (KEY_DELETE_WIN, KEY_DELETE_LINUX)
KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_PAGE_UP = 2162688
KEY_PAGE_DOWN = 2228224
KEY_HOME = 2359296
KEY_END = 2293760


def _is_yolo_suggestion(box: "Box") -> bool:
    """Sprawdza czy ramka pochodzi z podpowiedzi YOLO (niezatwierdzonej)."""
    return box.label.startswith("yolo:") and not box.manual


class Reviewer:
    """
    Główna klasa interfejsu graficznego.
    Obsługuje wyświetlanie obrazu, rysowanie nakładek (overlays), zdarzenia myszy
    oraz klawiatury (skróty klawiszowe i wprowadzanie tekstu).

    Podpowiedzi YOLO są wyróżnione żółtą przerywaną ramką.
    Zatwierdzenie (Enter lub klik) usuwa etykietę 'yolo:XX' i oznacza ramkę jako OK.
    """

    def __init__(
        self,
        image: np.ndarray,
        fname: str,
        existing_boxes: list[dict],
        target: Optional[int] = None,
        all_files: Optional[list[str]] = None,
        row_tolerance: int = 30,
        min_w: int = 30,
        min_h: int = 20,
        yolo_status: str = "",
    ) -> None:
        self.orig = image.copy()
        self.ih, self.iw = image.shape[:2]
        self.fname = fname
        self.target = target
        self.all_files = all_files or [fname]
        self.row_tolerance = row_tolerance
        self.min_w = min_w
        self.min_h = min_h
        self.yolo_status = yolo_status

        # Konwersja słowników na obiekty klasy Box
        self.boxes: list[Box] = []
        self.deleted_boxes: list[Box] = []
        for e in existing_boxes:
            if e.get("ok", True) and not e.get("deleted", False):
                b_rect = (e["x"], e["y"], e["w"], e["h"])
                if not any(iou(b_rect, box.rect) > 0.15 for box in self.boxes):
                    self.boxes.append(
                        Box(
                            e["x"],
                            e["y"],
                            e["w"],
                            e["h"],
                            ok=e.get("ok", True),
                            manual=e.get("manual", False),
                            label=e.get("label", ""),
                        )
                    )

        self.boxes.sort(key=lambda b: (b.y // self.row_tolerance, b.x))

        self.win_w, self.win_h = 1400, 800
        self.img_w = self.win_w - PANEL_W
        self.scale = 1.0
        self.offset = [0.0, 0.0]

        self._pan_start = None
        self._pan_off_start = None
        self._draw_start = None
        self._draw_cur = None
        self._selected: Optional[int] = None
        self.saved = False
        self._dirty = True

        self._list_active = False
        self._list_idx = 0
        try:
            self._list_idx = self.all_files.index(fname)
        except ValueError:
            pass
        self.requested_file: Optional[str] = None
        self._list_jump_buf: str = ""   # bufor cyfr do skoku po numerze

        def _get_font(size):
            for f in ["arial.ttf", "segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf"]:
                try:
                    return ImageFont.truetype(f, size)
                except:
                    continue
            return ImageFont.load_default()

        self.font_main = _get_font(16)
        self.font_large = _get_font(26)
        self.font_small = _get_font(12)

    # ------------------------------------------------------------------
    # Rysowanie tekstu (Unicode / ASCII)
    # ------------------------------------------------------------------

    def _draw_text(self, img, text, pos, font, color, anchor="la"):
        is_unicode = any(ord(c) > 127 for c in text)
        if not is_unicode:
            scale = 0.4
            if font == self.font_main:
                scale = 0.5
            elif font == self.font_large:
                scale = 0.8
            y_off = (
                12 if font == self.font_small
                else (16 if font == self.font_main else 26)
            )
            cv2.putText(img, text, (pos[0], pos[1] + y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]), anchor="la")
            img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _shorten(name: str, max_len: int = 25) -> str:
        if len(name) <= max_len:
            return name
        ext = name.split(".")[-1] if "." in name else ""
        part = (max_len - 5) // 2
        return f"{name[:part]}...{name[-(part+len(ext)+1):]}"

    def _fit(self, reset_view: bool = True):
        rect = cv2.getWindowImageRect(WINDOW)
        if rect[2] > 0 and rect[3] > 0:
            self.win_w, self.win_h = rect[2], rect[3]
            self.img_w = self.win_w - PANEL_W
        if reset_view:
            sx = self.img_w / self.iw
            sy = (self.win_h - 40) / self.ih
            self.scale = min(sx, sy, 1.0)
            self.offset = [0.0, 0.0]

    def _img2scr(self, px, py):
        return (int((px - self.offset[0]) * self.scale),
                int((py - self.offset[1]) * self.scale))

    def _scr2img(self, sx, sy):
        return (sx / self.scale + self.offset[0], sy / self.scale + self.offset[1])

    def _box_at(self, imgx, imgy):
        for i in range(len(self.boxes) - 1, -1, -1):
            b = self.boxes[i]
            if b.x <= imgx <= b.x + b.w and b.y <= imgy <= b.y + b.h:
                return i
        return None

    def _remove_box(self, idx: int):
        box = self.boxes.pop(idx)
        box.deleted = True
        self.deleted_boxes.append(box)
        if self._selected == idx:
            self._selected = None
        elif self._selected is not None and self._selected > idx:
            self._selected -= 1

    def _confirm_yolo(self, idx: int):
        """Zatwierdza podpowiedź YOLO: czyści etykietę i oznacza jako manual=True."""
        box = self.boxes[idx]
        if _is_yolo_suggestion(box):
            box.label = ""
            box.manual = True
            box.ok = True

    # ------------------------------------------------------------------
    # Renderowanie główne
    # ------------------------------------------------------------------

    def _render(self):
        self._fit(reset_view=False)
        vis_w = self.img_w / self.scale
        vis_h = (self.win_h - 40) / self.scale
        self.offset[0] = max(0.0, min(self.offset[0], max(0.0, self.iw - vis_w)))
        self.offset[1] = max(0.0, min(self.offset[1], max(0.0, self.ih - vis_h)))

        ox, oy = int(self.offset[0]), int(self.offset[1])
        x2 = min(ox + int(vis_w) + 1, self.iw)
        y2 = min(oy + int(vis_h) + 1, self.ih)
        crop = self.orig[oy:y2, ox:x2].copy()

        cw = int((x2 - ox) * self.scale)
        ch = int((y2 - oy) * self.scale)
        if cw < 1 or ch < 1:
            canvas = np.zeros((10, 10, 3), np.uint8)
        else:
            canvas = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_LINEAR)

        for i, box in enumerate(self.boxes):
            sx1 = int((box.x - ox) * self.scale)
            sy1 = int((box.y - oy) * self.scale)
            sx2 = int((box.x + box.w - ox) * self.scale)
            sy2 = int((box.y + box.h - oy) * self.scale)
            if sx2 < 0 or sy2 < 0 or sx1 > cw or sy1 > ch:
                continue

            is_yolo = _is_yolo_suggestion(box)

            if i == self._selected:
                color, thick = COL_SELECTED, 3
            elif is_yolo:
                color, thick = COL_YOLO, 2
            elif box.manual:
                color, thick = COL_MANUAL, (2 if box.ok else 1)
            elif box.ok:
                color, thick = COL_OK, 2
            else:
                color, thick = COL_REJECT, 1

            if is_yolo and i != self._selected:
                # Przerywana ramka dla podpowiedzi YOLO
                self._draw_dashed_rect(canvas, sx1, sy1, sx2, sy2, color, thick)
            else:
                cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), color, thick)

            if self.scale >= 0.3:
                label_text = box.label if box.label else str(i + 1)
                self._draw_text(
                    canvas,
                    label_text,
                    (sx1 + 2, sy1 - 18 if sy1 > 20 else sy1 + 2),
                    self.font_small,
                    color,
                )

        if self._draw_start and self._draw_cur:
            dx1 = int((min(self._draw_start[0], self._draw_cur[0]) - ox) * self.scale)
            dy1 = int((min(self._draw_start[1], self._draw_cur[1]) - oy) * self.scale)
            dx2 = int((max(self._draw_start[0], self._draw_cur[0]) - ox) * self.scale)
            dy2 = int((max(self._draw_start[1], self._draw_cur[1]) - oy) * self.scale)
            cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), COL_DRAW, 2)

        full = np.zeros((self.win_h, self.win_w, 3), np.uint8)
        full[:ch, :cw] = canvas
        panel = np.full((self.win_h, PANEL_W, 3), COL_PANEL, np.uint8)
        self._render_panel_into(panel)
        full[:, self.img_w:] = panel

        cv2.rectangle(full, (0, self.win_h - 36), (self.img_w, self.win_h), COL_STATUS, -1)
        ok_count = sum(1 for b in self.boxes if b.ok and not _is_yolo_suggestion(b))
        yolo_count = sum(1 for b in self.boxes if _is_yolo_suggestion(b))
        target_str = f"/{self.target}" if self.target is not None else ""
        yolo_str = f"  |  YOLO: {yolo_count} podp." if yolo_count > 0 else ""
        st = (f" Plik: {self._shorten(self.fname, 45)}"
              f"  |  OK={ok_count}{target_str} ({len(self.boxes)}){yolo_str}")
        self._draw_text(full, st, (10, self.win_h - 26), self.font_main, COL_TEXT)

        if self._list_active:
            self._render_file_list(full)
        return full

    @staticmethod
    def _draw_dashed_rect(img, x1, y1, x2, y2, color, thick, dash=8):
        """Rysuje przerywaną ramkę (dla podpowiedzi YOLO)."""
        pts = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
        for (ax, ay), (bx, by) in pts:
            dist = max(abs(bx - ax), abs(by - ay))
            steps = max(1, dist // (dash * 2))
            for s in range(steps):
                t0 = s / steps
                t1 = (s + 0.5) / steps
                p0 = (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0))
                p1 = (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1))
                cv2.line(img, p0, p1, color, thick)

    def _render_panel_into(self, panel):
        py = 35
        self._draw_text(panel, "PANEL EDYCJI", (20, py), self.font_large, COL_HIGHLIGHT)
        py += 35

        # Status YOLO w panelu
        if self.yolo_status:
            self._draw_text(panel, self.yolo_status, (20, py), self.font_small, COL_YOLO)
            py += 20
        py += 10

        if self._selected is not None:
            box = self.boxes[self._selected]
            is_yolo = _is_yolo_suggestion(box)
            label_color = COL_YOLO if is_yolo else (0, 255, 0)
            kind = "YOLO (niezatw.)" if is_yolo else f"Ramka nr {self._selected + 1}"

            self._draw_text(panel, f"Zaznaczono: {kind}", (20, py), self.font_main, COL_TEXT)
            py += 25

            cx1 = max(box.x - 15, 0)
            cy1 = max(box.y - 15, 0)
            cx2 = min(box.x + box.w + 15, self.iw)
            cy2 = min(box.y + box.h + 15, self.ih)
            z_crop = self.orig[cy1:cy2, cx1:cx2]
            if z_crop.size > 0:
                scale_z = min((PANEL_W - 40) / z_crop.shape[1], 150 / z_crop.shape[0])
                nh = int(z_crop.shape[0] * scale_z)
                nw = int(z_crop.shape[1] * scale_z)
                z_crop_r = cv2.resize(z_crop, (nw, nh))
                panel[py:py + nh, 20:20 + nw] = z_crop_r
                py += nh + 15

            display_label = box.label if not is_yolo else box.label
            self._draw_text(panel, display_label + "_", (30, py), self.font_large, label_color)
            py += 45

            if is_yolo:
                hints = [
                    ("[ENTER] - Zatwierdz YOLO", COL_YOLO),
                    ("[DEL]   - Odrzuc ramke", (120, 120, 255)),
                    ("[ESC]   - Odznacz", (180, 180, 180)),
                ]
            else:
                hints = [
                    ("[Klawisze] - pisanie", (180, 180, 180)),
                    ("[BS] - usun znak", (180, 180, 180)),
                    ("[DEL] - usun ramke", (120, 120, 255)),
                    ("[ENTER] - OK / Dalej", COL_DRAW),
                ]
            for txt, col in hints:
                self._draw_text(panel, txt, (20, py), self.font_small, col)
                py += 20
        else:
            help_lines = [
                ("[Enter] - Zapisz", COL_TEXT),
                ("[A] - Zatwierdz wszystko YOLO", COL_YOLO),
                ("[R] - Odrzuc wszystko YOLO", (120, 120, 255)),
                ("[L] - Lista plikow", COL_HIGHLIGHT),
                ("[1/2] OK/Odrzuc", (150, 150, 150)),
                ("[F/0] Dopasuj/100%", (150, 150, 150)),
                ("PPM: Nowa ramka", (150, 150, 150)),
                ("Drag LPM: Przesun", (150, 150, 150)),
            ]
            for txt, col in help_lines:
                self._draw_text(panel, txt, (20, py), self.font_small, col)
                py += 22

    def _render_file_list(self, full):
        lw, lh = 900, 560
        lx = (self.img_w - lw) // 2
        ly = (self.win_h - lh) // 2
        cv2.rectangle(full, (lx, ly), (lx + lw, ly + lh), (20, 20, 25), -1)
        cv2.rectangle(full, (lx, ly), (lx + lw, ly + lh), COL_HIGHLIGHT, 2)

        # Tytuł i licznik
        total = len(self.all_files)
        self._draw_text(full, f"WYBIERZ PLIK  ({self._list_idx + 1}/{total})",
                        (lx + 20, ly + 15), self.font_large, COL_HIGHLIGHT)

        # Lista plików (lewa część)
        list_w = 560
        visible = 12
        start = max(0, self._list_idx - visible // 2)
        end = min(total, start + visible)
        cur_py = ly + 60
        for i in range(start, end):
            is_cur = i == self._list_idx
            if is_cur:
                cv2.rectangle(full,
                              (lx + 10, cur_py - 4),
                              (lx + list_w, cur_py + 28),
                              (60, 60, 70), -1)
            self._draw_text(
                full,
                f"{i+1}. {self._shorten(self.all_files[i], 45)}",
                (lx + 25, cur_py),
                self.font_large if is_cur else self.font_main,
                COL_HIGHLIGHT if is_cur else COL_TEXT,
            )
            cur_py += 33

        # Separator pionowy
        cv2.line(full, (lx + list_w + 15, ly + 10), (lx + list_w + 15, ly + lh - 10),
                 (80, 80, 90), 1)

        # Legenda (prawa część)
        leg_x = lx + list_w + 30
        leg_y = ly + 55
        legend = [
            ("NAWIGACJA", COL_HIGHLIGHT),
            ("", COL_TEXT),
            ("[Gore/Dol]  - o 1 pozycje", COL_TEXT),
            ("[PgUp/PgDn] - o 10 pozycji", COL_TEXT),
            ("[Home]      - pierwszy plik", COL_TEXT),
            ("[End]       - ostatni plik", COL_TEXT),
            ("", COL_TEXT),
            ("SKOK DO NUMERU", COL_HIGHLIGHT),
            ("", COL_TEXT),
            ("Wpisz numer + Enter", COL_TEXT),
            ("(cyfry widoczne ponizej)", (150, 150, 150)),
            ("", COL_TEXT),
            ("INNE", COL_HIGHLIGHT),
            ("", COL_TEXT),
            ("[Enter] - otworz plik", COL_TEXT),
            ("[ESC/L] - zamknij liste", (150, 150, 150)),
        ]
        for txt, col in legend:
            if txt:
                self._draw_text(full, txt, (leg_x, leg_y), self.font_small, col)
            leg_y += 19

        # Pole wpisanego numeru
        if self._list_jump_buf:
            jump_txt = f"Skok do: {self._list_jump_buf}_"
            cv2.rectangle(full, (lx + 15, ly + lh - 45),
                          (lx + list_w, ly + lh - 10), (50, 50, 60), -1)
            self._draw_text(full, jump_txt, (lx + 25, ly + lh - 38),
                            self.font_main, (0, 255, 0))
        else:
            self._draw_text(full, "Wpisz numer aby skoczyc do pliku...",
                            (lx + 25, ly + lh - 35), self.font_small, (80, 80, 90))

    # ------------------------------------------------------------------
    # Obsługa myszy
    # ------------------------------------------------------------------

    def _mouse(self, event, sx, sy, flags, param):
        rect = cv2.getWindowImageRect(WINDOW)
        if rect[2] > 0 and rect[3] > 0:
            sx = int(sx * (self.win_w / rect[2]))
            sy = int(sy * (self.win_h / rect[3]))
        if self._list_active:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._list_active = False
                self._dirty = True
            return
        if sx > self.img_w:
            return
        imgx, imgy = self._scr2img(sx, sy)

        if event == cv2.EVENT_MOUSEWHEEL:
            new_scale = max(ZOOM_MIN,
                            min(ZOOM_MAX, self.scale * (1 + (1 if flags > 0 else -1) * ZOOM_STEP)))
            self.offset[0] += imgx * (1 - new_scale / self.scale)
            self.offset[1] += imgy * (1 - new_scale / self.scale)
            self.scale, self._dirty = new_scale, True

        elif event == cv2.EVENT_LBUTTONDOWN:
            self._pan_start, self._pan_off_start = (sx, sy), self.offset[:]

        elif event == cv2.EVENT_MOUSEMOVE and self._pan_start:
            self.offset[0] = self._pan_off_start[0] - (sx - self._pan_start[0]) / self.scale
            self.offset[1] = self._pan_off_start[1] - (sy - self._pan_start[1]) / self.scale
            self._dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            if (self._pan_start
                    and abs(sx - self._pan_start[0]) + abs(sy - self._pan_start[1]) <= 10):
                idx = self._box_at(imgx, imgy)
                if idx is not None:
                    if self._selected == idx:
                        # Klik na zaznaczoną: toggle ok / zatwierdź YOLO
                        if _is_yolo_suggestion(self.boxes[idx]):
                            self._confirm_yolo(idx)
                        else:
                            self.boxes[idx].ok = not self.boxes[idx].ok
                    else:
                        self._selected = idx
                else:
                    self._selected = None
            self._pan_start, self._dirty = None, True

        elif event == cv2.EVENT_RBUTTONDOWN:
            self._draw_start = self._draw_cur = (imgx, imgy)

        elif event == cv2.EVENT_MOUSEMOVE and self._draw_start:
            self._draw_cur, self._dirty = (imgx, imgy), True

        elif event == cv2.EVENT_RBUTTONUP and self._draw_start:
            x1 = int(min(self._draw_start[0], imgx))
            y1 = int(min(self._draw_start[1], imgy))
            x2 = int(max(self._draw_start[0], imgx))
            y2 = int(max(self._draw_start[1], imgy))
            if x2 - x1 >= self.min_w and y2 - y1 >= self.min_h:
                self.boxes.append(Box(x1, y1, x2 - x1, y2 - y1, manual=True))
                self._selected = len(self.boxes) - 1
            self._draw_start = self._draw_cur = None
            self._dirty = True

    # ------------------------------------------------------------------
    # Główna pętla
    # ------------------------------------------------------------------

    def run(self) -> bool:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, self.win_w, self.win_h)
        cv2.setMouseCallback(WINDOW, self._mouse)
        self._fit()
        while True:
            if self._dirty:
                cv2.imshow(WINDOW, self._render())
                self._dirty = False
            raw_key = cv2.waitKeyEx(15)
            if raw_key == -1:
                continue
            key = raw_key & 0xFF

            if self._list_active:
                n_files = len(self.all_files)
                if key == KEY_ESC or key in (ord("l"), ord("L")):
                    self._list_active = False
                    self._list_jump_buf = ""
                elif raw_key == KEY_UP:
                    self._list_idx = max(0, self._list_idx - 1)
                    self._list_jump_buf = ""
                elif raw_key == KEY_DOWN:
                    self._list_idx = min(n_files - 1, self._list_idx + 1)
                    self._list_jump_buf = ""
                elif raw_key == KEY_PAGE_UP:
                    self._list_idx = max(0, self._list_idx - 10)
                    self._list_jump_buf = ""
                elif raw_key == KEY_PAGE_DOWN:
                    self._list_idx = min(n_files - 1, self._list_idx + 10)
                    self._list_jump_buf = ""
                elif raw_key == KEY_HOME:
                    self._list_idx = 0
                    self._list_jump_buf = ""
                elif raw_key == KEY_END:
                    self._list_idx = n_files - 1
                    self._list_jump_buf = ""
                elif key in KEY_ENTER:
                    if self._list_jump_buf:
                        # Skok do wpisanego numeru
                        try:
                            target = int(self._list_jump_buf) - 1
                            self._list_idx = max(0, min(n_files - 1, target))
                        except ValueError:
                            pass
                        self._list_jump_buf = ""
                    else:
                        self.requested_file = self.all_files[self._list_idx]
                        self.saved = True
                        break
                elif key in KEY_BACKSPACE:
                    self._list_jump_buf = self._list_jump_buf[:-1]
                elif chr(key).isdigit():
                    self._list_jump_buf += chr(key)
                    # Podgląd na żywo: przesuń zaznaczenie
                    try:
                        target = int(self._list_jump_buf) - 1
                        if 0 <= target < n_files:
                            self._list_idx = target
                    except ValueError:
                        pass
                self._dirty = True
                continue

            if self._selected is not None:
                if key == KEY_ESC:
                    self._selected = None
                elif key in KEY_ENTER:
                    # Enter na YOLO = zatwierdź; na zwykłej = OK i przejdź dalej
                    if _is_yolo_suggestion(self.boxes[self._selected]):
                        self._confirm_yolo(self._selected)
                        self._selected = (
                            self._selected + 1
                            if self._selected + 1 < len(self.boxes) else None
                        )
                    else:
                        self.boxes[self._selected].ok = True
                        self._selected = (
                            self._selected + 1
                            if self._selected + 1 < len(self.boxes) else None
                        )
                elif raw_key in KEY_DELETE_CODES:
                    self._remove_box(self._selected)
                elif key in KEY_BACKSPACE:
                    if self.boxes[self._selected].label:
                        self.boxes[self._selected].label = self.boxes[self._selected].label[:-1]
                    else:
                        self._remove_box(self._selected)
                elif key == 22 or raw_key == 11534358:  # Ctrl+V
                    if _CLIPBOARD_OK:
                        try:
                            clipboard_text = pyperclip.paste()
                            if clipboard_text:
                                if _is_yolo_suggestion(self.boxes[self._selected]):
                                    self.boxes[self._selected].label = clipboard_text.strip()
                                    self.boxes[self._selected].manual = True
                                else:
                                    self.boxes[self._selected].label += clipboard_text.strip()
                        except Exception:
                            pass
                else:
                    char = self._decode_key(raw_key)
                    if char and char.isprintable():
                        # Pisanie nowej etykiety czyści oznaczenie YOLO
                        if _is_yolo_suggestion(self.boxes[self._selected]):
                            self.boxes[self._selected].label = char
                            self.boxes[self._selected].manual = True
                        else:
                            self.boxes[self._selected].label += char
                self._dirty = True
            else:
                if key in (KEY_ESC, ord("q"), ord("Q")):
                    break
                elif key in KEY_ENTER:
                    self.saved = True
                    break
                elif key in (ord("a"), ord("A")):
                    # Zatwierdź wszystkie podpowiedzi YOLO
                    count = 0
                    for idx in range(len(self.boxes)):
                        if _is_yolo_suggestion(self.boxes[idx]):
                            self._confirm_yolo(idx)
                            count += 1
                    print(f"  [GUI] Zatwierdzono {count} podpowiedzi YOLO.")
                    self._dirty = True
                elif key in (ord("r"), ord("R")):
                    # Odrzuć wszystkie podpowiedzi YOLO
                    to_remove = [i for i, b in enumerate(self.boxes) if _is_yolo_suggestion(b)]
                    for idx in reversed(to_remove):
                        self._remove_box(idx)
                    print(f"  [GUI] Odrzucono {len(to_remove)} podpowiedzi YOLO.")
                    self._dirty = True
                elif key in (ord("l"), ord("L")):
                    self._list_active = True
                    self._dirty = True
                elif key in (ord("1"), ord("!")):
                    [setattr(b, "ok", True) for b in self.boxes]
                    self._dirty = True
                elif key in (ord("2"), ord("@")):
                    [setattr(b, "ok", False) for b in self.boxes]
                    self._dirty = True
                elif key in (ord("f"), ord("F")):
                    self._fit(True)
                    self._dirty = True
                elif key in (ord("0"), ord(")")):
                    self.scale, self.offset = 1.0, [0.0, 0.0]
                    self._dirty = True

        cv2.destroyAllWindows()
        return self.saved

    @staticmethod
    def _decode_key(raw_key: int) -> str:
        pl_map = {
            185: "ą", 230: "ć", 234: "ę", 179: "ł", 241: "ń",
            243: "ó", 156: "ś", 159: "ź", 191: "ż",
            161: "Ą", 198: "Ć", 202: "Ę", 163: "Ł", 209: "Ń",
            211: "Ó", 140: "Ś", 143: "Ź", 175: "Ż",
        }
        if raw_key in pl_map:
            return pl_map[raw_key]
        if raw_key < 256:
            try:
                return bytes([raw_key]).decode("cp1250")
            except:
                pass
        try:
            return chr(raw_key)
        except:
            return ""

    def get_result(self) -> list[dict]:
        return [b.to_dict() for b in self.boxes] + [b.to_dict() for b in self.deleted_boxes]