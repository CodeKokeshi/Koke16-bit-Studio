from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from daw.instruments import INSTRUMENT_LIBRARY
from daw.generator import GENRE_NAMES
from daw.shortcuts import (
    KeyBinding,
    ShortcutConfig,
    chord_to_binding,
    key_to_text,
    modifier_from_key,
    modifier_to_text,
)


class AddTrackDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Piano Roll Track")
        self.resize(420, 360)

        layout = QVBoxLayout(self)

        title = QLabel("Choose an instrument")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(title)

        self.list_widget = QListWidget()
        for inst in INSTRUMENT_LIBRARY:
            item = QListWidgetItem(f"{inst['name']}  ·  {inst['family']}")
            item.setData(Qt.ItemDataRole.UserRole, inst)
            self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(0)
        self.list_widget.itemDoubleClicked.connect(lambda _: self.accept())
        layout.addWidget(self.list_widget, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_instrument(self) -> dict[str, str] | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)


class AudioToMusicDialog(QDialog):
    def __init__(self, track_names: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Audio → Music")
        self.resize(460, 460)

        layout = QVBoxLayout(self)

        title = QLabel("Choose conversion mode + destination")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(title)

        self.chk_auto_multitrack = QCheckBox("Auto-retrofy to multiple tracks (Lead/Bass/Harmony/Drums)")
        self.chk_auto_multitrack.setChecked(True)
        layout.addWidget(self.chk_auto_multitrack)

        self.chk_auto_select_instruments = QCheckBox("Auto-select instruments for generated tracks")
        self.chk_auto_select_instruments.setChecked(True)
        layout.addWidget(self.chk_auto_select_instruments)

        self.lbl_single_instr = QLabel("Single-track instrument (used when auto-multitrack is OFF):")
        self.lbl_single_instr.setStyleSheet("color:#9a9a9a; font-size:11px;")
        layout.addWidget(self.lbl_single_instr)

        self.list_widget = QListWidget()
        for inst in INSTRUMENT_LIBRARY:
            item = QListWidgetItem(f"{inst['name']}  ·  {inst['family']}")
            item.setData(Qt.ItemDataRole.UserRole, inst)
            self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(0)
        layout.addWidget(self.list_widget, 1)

        self.radio_new = QRadioButton("Create new track")
        self.radio_overwrite = QRadioButton("Write to existing track")
        self.radio_new.setChecked(True)
        layout.addWidget(self.radio_new)
        layout.addWidget(self.radio_overwrite)

        row = QHBoxLayout()
        row.addWidget(QLabel("Target Track"))
        self.combo_tracks = QComboBox()
        self.combo_tracks.addItems(track_names)
        self.combo_tracks.setEnabled(False)
        row.addWidget(self.combo_tracks, 1)
        layout.addLayout(row)

        self.chk_auto_multitrack.toggled.connect(self._update_destination_controls)
        self.chk_auto_select_instruments.toggled.connect(self._update_destination_controls)
        self.radio_new.toggled.connect(self._update_destination_controls)
        self.radio_overwrite.toggled.connect(self._update_destination_controls)
        self._update_destination_controls()

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _update_destination_controls(self):
        allow_overwrite_target = self.radio_overwrite.isChecked() and not self.chk_auto_multitrack.isChecked()
        self.combo_tracks.setEnabled(allow_overwrite_target)

        if self.chk_auto_multitrack.isChecked():
            self.radio_new.setChecked(True)
            self.radio_overwrite.setEnabled(False)
            self.chk_auto_select_instruments.setEnabled(True)
            self.list_widget.setEnabled(False)
            self.lbl_single_instr.setEnabled(False)
        else:
            self.radio_overwrite.setEnabled(True)
            self.chk_auto_select_instruments.setEnabled(False)
            self.list_widget.setEnabled(True)
            self.lbl_single_instr.setEnabled(True)

    def selected_instrument(self) -> dict[str, str] | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def create_new_track(self) -> bool:
        return self.radio_new.isChecked()

    def target_track_name(self) -> str:
        return self.combo_tracks.currentText()

    def auto_multitrack(self) -> bool:
        return self.chk_auto_multitrack.isChecked()

    def auto_select_instruments(self) -> bool:
        return self.chk_auto_select_instruments.isChecked()


class BeautifyDialog(QDialog):
    """Ask the user whether to beautify the current track or all tracks."""

    def __init__(self, current_track_name: str | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("✨ Beautify Tracks")
        self.resize(430, 260)

        layout = QVBoxLayout(self)

        title = QLabel("Apply music-theory beautification")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(title)

        desc = QLabel(
            "Analyses each track's role (lead / bass / harmony / drums) "
            "from its pitch range, instrument, and note patterns, then "
            "applies role-appropriate music theory corrections."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #9a9a9a; font-size: 11px;")
        layout.addWidget(desc)

        self.radio_current = QRadioButton(
            f"Current track only ({current_track_name})" if current_track_name
            else "Current track only"
        )
        self.radio_all = QRadioButton("All tracks")
        self.radio_current.setChecked(True)
        if not current_track_name:
            self.radio_current.setEnabled(False)
            self.radio_all.setChecked(True)

        self._target_group = QButtonGroup(self)
        self._target_group.setExclusive(True)
        self._target_group.addButton(self.radio_current)
        self._target_group.addButton(self.radio_all)

        layout.addWidget(self.radio_current)
        layout.addWidget(self.radio_all)

        mode_title = QLabel("Playback intent")
        mode_title.setStyleSheet("font-size: 13px; font-weight: 600; margin-top: 6px;")
        layout.addWidget(mode_title)

        self.radio_standard = QRadioButton("🎬 Standard beautify (one-time playback)")
        self.radio_loop = QRadioButton("🔁 Loop-aware beautify (preserve seamless restart)")
        self.radio_standard.setChecked(True)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self.radio_standard)
        self._mode_group.addButton(self.radio_loop)

        layout.addWidget(self.radio_standard)
        layout.addWidget(self.radio_loop)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def apply_to_all(self) -> bool:
        return self.radio_all.isChecked()

    def loop_aware(self) -> bool:
        return self.radio_loop.isChecked()


class RoleInstrumentDialog(QDialog):
    def __init__(self, role_name: str, default_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Choose Instrument: {role_name}")
        self.resize(420, 360)

        layout = QVBoxLayout(self)
        title = QLabel(f"Choose instrument for {role_name}")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(title)

        self.list_widget = QListWidget()
        default_index = 0
        for idx, inst in enumerate(INSTRUMENT_LIBRARY):
            item = QListWidgetItem(f"{inst['name']}  ·  {inst['family']}")
            item.setData(Qt.ItemDataRole.UserRole, inst)
            self.list_widget.addItem(item)
            if inst["name"] == default_name:
                default_index = idx

        self.list_widget.setCurrentRow(default_index)
        self.list_widget.itemDoubleClicked.connect(lambda _: self.accept())
        layout.addWidget(self.list_widget, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_instrument(self) -> dict[str, str] | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)


class ListeningField(QLineEdit):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFixedWidth(200)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def mousePressEvent(self, event):  # noqa: N802
        self.setFocus()
        self.clicked.emit()
        super().mousePressEvent(event)


class ShortcutSettingsDialog(QDialog):
    def __init__(self, config: ShortcutConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(640, 460)
        self._config = ShortcutConfig(
            modifier=config.modifier,
            controller=config.controller,
            box_select=KeyBinding(config.box_select.modifiers, config.box_select.key),
            zoom=KeyBinding(config.zoom.modifiers, config.zoom.key),
            undo=KeyBinding(config.undo.modifiers, config.undo.key),
            redo_primary=KeyBinding(config.redo_primary.modifiers, config.redo_primary.key),
            redo_secondary=KeyBinding(config.redo_secondary.modifiers, config.redo_secondary.key),
            select_all_notes=KeyBinding(config.select_all_notes.modifiers, config.select_all_notes.key),
            delete_primary=KeyBinding(config.delete_primary.modifiers, config.delete_primary.key),
            delete_secondary=KeyBinding(config.delete_secondary.modifiers, config.delete_secondary.key),
        )
        self._active_field_name: str | None = None
        self._active_field: ListeningField | None = None
        self._captured_keys: list[int] = []
        self._pressed_keys: set[int] = set()
        self._capture_started = False
        self._listen_seconds = 0

        self._listen_timer = QTimer(self)
        self._listen_timer.setInterval(1000)
        self._listen_timer.timeout.connect(self._on_listen_tick)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title = QLabel("Shortcut Controls")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        root.addWidget(title)

        self.lbl_hint = QLabel("Click a field to listen for 3 seconds. Capture ends when all pressed keys are released.")
        self.lbl_hint.setStyleSheet("color:#8f8f8f;")
        root.addWidget(self.lbl_hint)

        rows_root = QVBoxLayout()
        rows_root.setSpacing(8)
        root.addLayout(rows_root, 1)

        rows_grid = QGridLayout()
        rows_grid.setHorizontalSpacing(10)
        rows_grid.setVerticalSpacing(8)
        rows_grid.setColumnMinimumWidth(0, 130)
        rows_grid.setColumnMinimumWidth(1, 200)
        rows_grid.setColumnMinimumWidth(2, 170)
        rows_grid.setColumnStretch(2, 1)
        rows_root.addLayout(rows_grid)

        self.f_modifier = ListeningField()
        self.f_box_select = ListeningField()
        self.f_zoom = ListeningField()
        self.f_controller = ListeningField()
        self.f_undo = ListeningField()
        self.f_redo_primary = ListeningField()
        self.f_redo_secondary = ListeningField()
        self.f_select_all = ListeningField()
        self.f_delete_primary = ListeningField()
        self.f_delete_secondary = ListeningField()

        self.lbl_box_select = QLabel("+ Left Click Drag")
        self.lbl_zoom = QLabel("+ Scroll / Mouse Wheel")

        self._add_settings_row(rows_grid, 0, "Modifier", self.f_modifier, "")
        self._add_settings_row(rows_grid, 1, "Box Select Notes", self.f_box_select, "+ Left Click Drag")
        self._add_settings_row(rows_grid, 2, "Zoom Piano Roll", self.f_zoom, "+ Scroll / Mouse Wheel")
        self._add_settings_row(rows_grid, 3, "Controller", self.f_controller, "")
        self._add_settings_row(rows_grid, 4, "Undo", self.f_undo, "")
        self._add_settings_row(rows_grid, 5, "Redo (Primary)", self.f_redo_primary, "")
        self._add_settings_row(rows_grid, 6, "Redo (Alternate)", self.f_redo_secondary, "")
        self._add_settings_row(rows_grid, 7, "Select All Notes", self.f_select_all, "")
        self._add_settings_row(rows_grid, 8, "Delete Notes", self.f_delete_primary, "")
        self._add_settings_row(rows_grid, 9, "Delete Notes (Alt)", self.f_delete_secondary, "")
        rows_root.addStretch(1)

        self.f_modifier.clicked.connect(lambda: self._begin_listen("modifier", self.f_modifier))
        self.f_box_select.clicked.connect(lambda: self._begin_listen("box_select", self.f_box_select))
        self.f_zoom.clicked.connect(lambda: self._begin_listen("zoom", self.f_zoom))
        self.f_controller.clicked.connect(lambda: self._begin_listen("controller", self.f_controller))
        self.f_undo.clicked.connect(lambda: self._begin_listen("undo", self.f_undo))
        self.f_redo_primary.clicked.connect(lambda: self._begin_listen("redo_primary", self.f_redo_primary))
        self.f_redo_secondary.clicked.connect(lambda: self._begin_listen("redo_secondary", self.f_redo_secondary))
        self.f_select_all.clicked.connect(lambda: self._begin_listen("select_all_notes", self.f_select_all))
        self.f_delete_primary.clicked.connect(lambda: self._begin_listen("delete_primary", self.f_delete_primary))
        self.f_delete_secondary.clicked.connect(lambda: self._begin_listen("delete_secondary", self.f_delete_secondary))

        self._refresh_labels()

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def shortcut_config(self) -> ShortcutConfig:
        return ShortcutConfig(
            modifier=self._config.modifier,
            controller=self._config.controller,
            box_select=KeyBinding(self._config.box_select.modifiers, self._config.box_select.key),
            zoom=KeyBinding(self._config.zoom.modifiers, self._config.zoom.key),
            undo=KeyBinding(self._config.undo.modifiers, self._config.undo.key),
            redo_primary=KeyBinding(self._config.redo_primary.modifiers, self._config.redo_primary.key),
            redo_secondary=KeyBinding(self._config.redo_secondary.modifiers, self._config.redo_secondary.key),
            select_all_notes=KeyBinding(self._config.select_all_notes.modifiers, self._config.select_all_notes.key),
            delete_primary=KeyBinding(self._config.delete_primary.modifiers, self._config.delete_primary.key),
            delete_secondary=KeyBinding(self._config.delete_secondary.modifiers, self._config.delete_secondary.key),
        )

    def _add_settings_row(self, grid: QGridLayout, row: int, name: str, control: QWidget, suffix: str):
        lbl_name = QLabel(name)
        lbl_suffix = QLabel(suffix)
        lbl_suffix.setStyleSheet("color:#e7e7e7;")

        grid.addWidget(lbl_name, row, 0)
        grid.addWidget(control, row, 1)
        grid.addWidget(lbl_suffix, row, 2)

    def _field_with_tail(self, field: ListeningField, tail_label: QLabel) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(field)
        row.addWidget(tail_label)
        return host

    def _begin_listen(self, field_name: str, field: ListeningField):
        if self._active_field is field and self._listen_timer.isActive():
            return
        self._cancel_listen()
        self._active_field_name = field_name
        self._active_field = field
        self._captured_keys = []
        self._pressed_keys = set()
        self._capture_started = False
        self._listen_seconds = 3
        self._active_field.setText("Listening… 3")
        self._listen_timer.start()
        self.grabKeyboard()

    def _on_listen_tick(self):
        if self._active_field is None:
            self._cancel_listen()
            return
        if self._capture_started:
            return
        self._listen_seconds -= 1
        if self._listen_seconds <= 0:
            self._cancel_listen()
            self._refresh_labels()
            return
        self._active_field.setText(f"Listening… {self._listen_seconds}")

    def _cancel_listen(self):
        if self._listen_timer.isActive():
            self._listen_timer.stop()
        self._active_field_name = None
        self._active_field = None
        self._captured_keys = []
        self._pressed_keys = set()
        self._capture_started = False
        try:
            self.releaseKeyboard()
        except Exception:
            pass

    def keyPressEvent(self, event):  # noqa: N802
        if self._active_field_name is None:
            super().keyPressEvent(event)
            return
        key = int(event.key())

        if key not in self._captured_keys and len(self._captured_keys) < 4:
            self._captured_keys.append(key)
        self._pressed_keys.add(key)
        self._capture_started = True
        if self._listen_timer.isActive():
            self._listen_timer.stop()

        if self._active_field is not None:
            self._active_field.setText(self._keys_to_text(self._captured_keys) + "  (release to apply)")
        event.accept()

    def keyReleaseEvent(self, event):  # noqa: N802
        if self._active_field_name is None:
            super().keyReleaseEvent(event)
            return
        key = int(event.key())
        self._pressed_keys.discard(key)
        if self._capture_started and not self._pressed_keys:
            self._apply_captured_keys()
            self._cancel_listen()
            self._refresh_labels()
        event.accept()

    def _keys_to_text(self, keys: list[int]) -> str:
        if not keys:
            return "None"
        return " + ".join(key_to_text(k) for k in keys)

    def _modifiers_from_keys(self, keys: list[int]) -> Qt.KeyboardModifier:
        mods = Qt.KeyboardModifier.NoModifier
        for key in keys:
            mod = modifier_from_key(key)
            if mod is not None:
                mods |= mod
        return mods

    def _apply_captured_keys(self):
        if not self._captured_keys or self._active_field_name is None:
            return

        name = self._active_field_name
        if name in ("modifier", "controller"):
            mods = self._modifiers_from_keys(self._captured_keys)
            if mods != Qt.KeyboardModifier.NoModifier:
                setattr(self._config, name, mods)
            return

        binding = chord_to_binding(self._captured_keys)
        if name in ("box_select", "zoom"):
            if binding.modifiers == Qt.KeyboardModifier.NoModifier and binding.key == int(Qt.Key.Key_unknown):
                return
            setattr(self._config, name, binding)
            return

        if binding.key == int(Qt.Key.Key_unknown):
            return
        setattr(self._config, name, binding)

    def _format_binding_symbolic(self, binding: KeyBinding) -> str:
        parts = []
        if binding.modifiers == self._config.controller and binding.modifiers != Qt.KeyboardModifier.NoModifier:
            parts.append("CONTROLLER")
        elif binding.modifiers == self._config.modifier and binding.modifiers != Qt.KeyboardModifier.NoModifier:
            parts.append("MODIFIER")
        elif (
            binding.modifiers == (self._config.controller | self._config.modifier)
            and self._config.controller != Qt.KeyboardModifier.NoModifier
            and self._config.modifier != Qt.KeyboardModifier.NoModifier
        ):
            parts.extend(["CONTROLLER", "MODIFIER"])
        else:
            mod_text = modifier_to_text(binding.modifiers)
            if mod_text != "None":
                parts.append(mod_text)

        if binding.key != int(Qt.Key.Key_unknown):
            parts.append(key_to_text(binding.key))

        return " + ".join(parts) if parts else "None"

    def _refresh_labels(self):
        if self._active_field_name is None:
            self._set_field_text(self.f_modifier, modifier_to_text(self._config.modifier))
            self._set_field_text(self.f_box_select, self._format_binding_symbolic(self._config.box_select))
            self._set_field_text(self.f_zoom, self._format_binding_symbolic(self._config.zoom))
            self._set_field_text(self.f_controller, modifier_to_text(self._config.controller))
            self._set_field_text(self.f_undo, self._format_binding_symbolic(self._config.undo))
            self._set_field_text(self.f_redo_primary, self._format_binding_symbolic(self._config.redo_primary))
            self._set_field_text(self.f_redo_secondary, self._format_binding_symbolic(self._config.redo_secondary))
            self._set_field_text(self.f_select_all, self._format_binding_symbolic(self._config.select_all_notes))
            self._set_field_text(self.f_delete_primary, self._format_binding_symbolic(self._config.delete_primary))
            self._set_field_text(self.f_delete_secondary, self._format_binding_symbolic(self._config.delete_secondary))

    def _set_field_text(self, field: ListeningField, text: str):
        field.setText(text)
        field.setCursorPosition(0)


class HelpDialog(QDialog):
    def __init__(self, config: ShortcutConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(560, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        shortcuts_tab = QWidget()
        shortcuts_layout = QVBoxLayout(shortcuts_tab)
        shortcuts_layout.setContentsMargins(10, 10, 10, 10)
        shortcuts_layout.setSpacing(8)

        title = QLabel("Shortcuts")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        shortcuts_layout.addWidget(title)

        rows = [
            ("Modifier", modifier_to_text(config.modifier)),
            ("Box Select Notes", f"{self._fmt(config.box_select, config)} + Left Click Drag"),
            ("Zoom Piano Roll", f"{self._fmt(config.zoom, config)} + Scroll / Mouse Wheel"),
            ("Controller", modifier_to_text(config.controller)),
            ("Undo", self._fmt(config.undo, config)),
            ("Redo", f"{self._fmt(config.redo_primary, config)} / {self._fmt(config.redo_secondary, config)}"),
            ("Select All Notes", self._fmt(config.select_all_notes, config)),
            ("Delete Notes", f"{self._fmt(config.delete_primary, config)} / {self._fmt(config.delete_secondary, config)}"),
        ]

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 150)
        grid.setColumnMinimumWidth(1, 260)
        grid.setColumnStretch(1, 1)

        for index, (action, shortcut) in enumerate(rows):
            lbl_action = QLabel(action)
            lbl_shortcut = QLabel(shortcut)
            lbl_shortcut.setStyleSheet("color:#00ffc8;")
            grid.addWidget(lbl_action, index, 0)
            grid.addWidget(lbl_shortcut, index, 1)

        shortcuts_layout.addLayout(grid)

        shortcuts_layout.addStretch(1)
        tabs.addTab(shortcuts_tab, "Shortcuts")

        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        about_layout.setContentsMargins(10, 10, 10, 10)
        about_layout.setSpacing(8)
        lbl_about = QLabel("Developed by CodeKokeshi, February 2026")
        lbl_about.setStyleSheet("font-size: 14px; color:#d7d7d7;")
        about_layout.addWidget(lbl_about)
        about_layout.addStretch(1)
        tabs.addTab(about_tab, "About")

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)

    def _fmt(self, binding: KeyBinding, config: ShortcutConfig) -> str:
        parts = []
        if binding.modifiers == config.controller and binding.modifiers != Qt.KeyboardModifier.NoModifier:
            parts.append("CONTROLLER")
        elif binding.modifiers == config.modifier and binding.modifiers != Qt.KeyboardModifier.NoModifier:
            parts.append("MODIFIER")
        elif (
            binding.modifiers == (config.controller | config.modifier)
            and config.controller != Qt.KeyboardModifier.NoModifier
            and config.modifier != Qt.KeyboardModifier.NoModifier
        ):
            parts.extend(["CONTROLLER", "MODIFIER"])
        else:
            mod_text = modifier_to_text(binding.modifiers)
            if mod_text != "None":
                parts.append(mod_text)

        if binding.key != int(Qt.Key.Key_unknown):
            parts.append(key_to_text(binding.key))

        return " + ".join(parts) if parts else "None"

class GenerateMusicDialog(QDialog):
    """Modal dialog to select a genre/mood for procedural music generation.

    Uses a two-click Category → Variant hierarchy so the list stays
    manageable even with 60+ genres.
    """

    # ── Category definitions ──────────────────────────────────────
    _CATEGORIES: list[tuple[str, str]] = [
        ("Town",        "🏘️"),
        ("Cave",        "🕳️"),
        ("HM Town",     "🌾"),
        ("Zelda Town",  "🧝"),
        ("Dungeon",     "🏚️"),
        ("Combat",      "⚔️"),
        ("Encounter",   "❗"),
        ("Boss Battle", "👹"),
        ("Overworld",   "🗺️"),
        ("Victory",     "🏆"),
    ]
    _PLATFORMS: list[tuple[str, str]] = [
        ("Generic",  "🎵"),
        ("NES",      "🎮"),
        ("Gameboy",  "🕹️"),
        ("GBA",      "📱"),
        ("SNES",     "🖥️"),
        ("Mix",      "🎲"),
    ]

    @staticmethod
    def _genre_key(platform: str, category: str) -> str:
        """Build the genre dict key from platform + category names."""
        # Town & Cave in generator.py use 'Platform Category' (no dash).
        if category in ("Town", "Cave"):
            return f"{platform} {category}"
        return f"{platform} - {category}"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎵 Generate Music")
        self.resize(560, 520)

        layout = QVBoxLayout(self)

        title = QLabel("Choose a genre / mood")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(title)

        desc = QLabel(
            "Pick a category on the left, then choose a platform\n"
            "variant on the right and click Generate."
        )
        desc.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        layout.addWidget(desc)

        # ── Two-panel genre selector ──────────────────────────────
        panels = QHBoxLayout()

        # Left panel – categories
        left_vbox = QVBoxLayout()
        cat_label = QLabel("Category")
        cat_label.setStyleSheet("font-size: 12px; font-weight: 600;")
        left_vbox.addWidget(cat_label)
        self.category_list = QListWidget()
        self.category_list.setFixedWidth(180)
        for cat_name, cat_icon in self._CATEGORIES:
            item = QListWidgetItem(f"{cat_icon}  {cat_name}")
            item.setData(Qt.ItemDataRole.UserRole, cat_name)
            self.category_list.addItem(item)
        left_vbox.addWidget(self.category_list)
        panels.addLayout(left_vbox)

        # Right panel – platform variants
        right_vbox = QVBoxLayout()
        var_label = QLabel("Platform variant")
        var_label.setStyleSheet("font-size: 12px; font-weight: 600;")
        right_vbox.addWidget(var_label)
        self.variant_list = QListWidget()
        right_vbox.addWidget(self.variant_list)
        panels.addLayout(right_vbox, 1)

        layout.addLayout(panels, 1)

        # Wire up selection
        self.category_list.currentItemChanged.connect(self._on_category_changed)
        self.variant_list.currentItemChanged.connect(self._on_variant_changed)
        self.variant_list.itemDoubleClicked.connect(lambda _: self.accept())

        # Select first category to populate the right panel
        self.category_list.setCurrentRow(0)

        # ── Playback mode ─────────────────────────────────────────
        mode_label = QLabel("Playback mode")
        mode_label.setStyleSheet("font-size: 13px; font-weight: 600; margin-top: 6px;")
        layout.addWidget(mode_label)

        self.radio_loop = QRadioButton(
            "🔁  Seamless loop  (8/16/32 bars — designed to repeat infinitely)"
        )
        self.radio_onetime = QRadioButton(
            "🎬  One-time play  (longer piece, natural ending)"
        )
        self.radio_loop.setChecked(True)
        layout.addWidget(self.radio_loop)
        layout.addWidget(self.radio_onetime)

        # ── Length selector ────────────────────────────────────────
        length_row = QHBoxLayout()
        length_lbl = QLabel("Length:")
        length_lbl.setStyleSheet("font-size: 12px;")
        self.combo_length = QComboBox()
        self._loop_lengths = [
            ("Short (8 bars)",  8),
            ("Medium (16 bars)", 16),
            ("Long (32 bars)", 32),
        ]
        self._onetime_lengths = [
            ("Medium (16 bars)", 16),
            ("Long (32 bars)",   32),
            ("Epic (48 bars)",   48),
            ("Marathon (64 bars)", 64),
        ]
        self._update_length_choices()
        default_idx = self.combo_length.findData(16)
        if default_idx >= 0:
            self.combo_length.setCurrentIndex(default_idx)
        self.radio_loop.toggled.connect(self._update_length_choices)
        length_row.addWidget(length_lbl)
        length_row.addWidget(self.combo_length, 1)
        layout.addLayout(length_row)

        # ── Track density selector ─────────────────────────────────
        density_row = QHBoxLayout()
        density_lbl = QLabel("Track density:")
        density_lbl.setStyleSheet("font-size: 12px;")
        self.combo_density = QComboBox()
        self._density_choices = [
            ("Minimal (3 tracks — Lead, Bass, Drums)",  "minimal"),
            ("Standard (4 tracks — Lead, Bass, Harmony, Drums)", "standard"),
            ("Rich (5-6 tracks — core + 1-2 extras)",  "rich"),
            ("Full (all tracks — 6-8, can be noisy)",   "full"),
        ]
        for label, value in self._density_choices:
            self.combo_density.addItem(label, value)
        # Default to "standard" (4 tracks — balanced mix)
        default_density = self.combo_density.findData("standard")
        if default_density >= 0:
            self.combo_density.setCurrentIndex(default_density)
        density_row.addWidget(density_lbl)
        density_row.addWidget(self.combo_density, 1)
        layout.addLayout(density_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Generate")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ── Category → populate variants ──────────────────────────────
    def _on_category_changed(self, current, _previous):
        """Populate the variant list when a category is selected."""
        self.variant_list.clear()
        if current is None:
            return
        category = current.data(Qt.ItemDataRole.UserRole)
        for plat_name, plat_icon in self._PLATFORMS:
            genre_key = self._genre_key(plat_name, category)
            if genre_key not in GENRE_NAMES:
                continue
            item = QListWidgetItem(f"{plat_icon}  {plat_name} — {category}")
            item.setData(Qt.ItemDataRole.UserRole, genre_key)
            self.variant_list.addItem(item)
        if self.variant_list.count():
            self.variant_list.setCurrentRow(0)

    # ── Smart defaults when variant changes ───────────────────────
    def _on_variant_changed(self, current, _previous):
        """Apply sensible defaults per genre when user switches selection."""
        if current is None:
            return
        genre = current.data(Qt.ItemDataRole.UserRole)
        # All Town variants default to seamless 16-bar loops.
        if "Town" in genre:
            if not self.radio_loop.isChecked():
                self.radio_loop.setChecked(True)
            idx = self.combo_length.findData(16)
            if idx >= 0:
                self.combo_length.setCurrentIndex(idx)
        # Cave / Dungeon default to seamless 32-bar loops.
        elif "Cave" in genre or "Dungeon" in genre:
            if not self.radio_loop.isChecked():
                self.radio_loop.setChecked(True)
            idx = self.combo_length.findData(32)
            if idx >= 0:
                self.combo_length.setCurrentIndex(idx)
        # Combat / Boss Battle / Overworld default to 32-bar loops.
        elif "Combat" in genre or "Boss" in genre or "Overworld" in genre:
            if not self.radio_loop.isChecked():
                self.radio_loop.setChecked(True)
            idx = self.combo_length.findData(32)
            if idx >= 0:
                self.combo_length.setCurrentIndex(idx)
        # Encounter default to shorter 16-bar loops.
        elif "Encounter" in genre:
            if not self.radio_loop.isChecked():
                self.radio_loop.setChecked(True)
            idx = self.combo_length.findData(16)
            if idx >= 0:
                self.combo_length.setCurrentIndex(idx)
        # Victory/Fanfare default to short 8-bar.
        elif "Victory" in genre:
            if not self.radio_loop.isChecked():
                self.radio_loop.setChecked(True)
            idx = self.combo_length.findData(8)
            if idx >= 0:
                self.combo_length.setCurrentIndex(idx)

    def _update_length_choices(self):
        self.combo_length.clear()
        choices = self._loop_lengths if self.radio_loop.isChecked() else self._onetime_lengths
        for label, value in choices:
            self.combo_length.addItem(label, value)
        idx = self.combo_length.findData(16)
        if idx >= 0:
            self.combo_length.setCurrentIndex(idx)

    def is_loop_mode(self) -> bool:
        return self.radio_loop.isChecked()

    def selected_bars(self) -> int:
        return self.combo_length.currentData() or 8

    def selected_density(self) -> str:
        return self.combo_density.currentData() or "standard"

    def selected_genre(self) -> str | None:
        item = self.variant_list.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None