from __future__ import annotations

from collections.abc import Callable
import tkinter as tk

import customtkinter as ctk


BG = "#090D18"
SURFACE = "#0F1628"
CARD = "#131C30"
CARD_HOVER = "#18233A"
INPUT = "#0B1222"
BORDER = "#25314B"
TEXT = "#F8FAFC"
MUTED = "#91A0B8"
SUBTLE = "#64748B"
ACCENT = "#7C5CFC"
ACCENT_HOVER = "#9278FF"
ACCENT_SOFT = "#2B2355"
SUCCESS = "#2DD4BF"
DANGER = "#FB7185"


def _font(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family="Segoe UI", size=size, weight=weight)


def _label(parent: tk.Misc, text: str = "", **kwargs: object) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        parent,
        text=text,
        text_color=kwargs.pop("text_color", MUTED),
        font=kwargs.pop("font", _font(13)),
        anchor=kwargs.pop("anchor", "w"),
        justify=kwargs.pop("justify", "left"),
        **kwargs,
    )


def _card(parent: tk.Misc, *, title: str, subtitle: str = "") -> tuple[ctk.CTkFrame, ctk.CTkFrame]:
    frame = ctk.CTkFrame(
        parent,
        fg_color=CARD,
        corner_radius=18,
        border_width=1,
        border_color=BORDER,
    )
    header = ctk.CTkFrame(frame, fg_color="transparent")
    header.pack(fill="x", padx=22, pady=(20, 0))
    _label(header, title, text_color=TEXT, font=_font(17, "bold")).pack(anchor="w")
    if subtitle:
        _label(header, subtitle, wraplength=620).pack(anchor="w", pady=(4, 0))
    body = ctk.CTkFrame(frame, fg_color="transparent")
    body.pack(fill="both", expand=True, padx=22, pady=(18, 22))
    return frame, body


def _button(parent: tk.Misc, text: str, command: Callable[[], None], *, primary: bool = False,
            width: int = 120, state: str = "normal") -> ctk.CTkButton:
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        width=width,
        height=40,
        corner_radius=10,
        fg_color=ACCENT if primary else CARD_HOVER,
        hover_color=ACCENT_HOVER if primary else BORDER,
        border_width=0 if primary else 1,
        border_color=BORDER,
        text_color=TEXT,
        text_color_disabled=SUBTLE,
        font=_font(13, "bold"),
        state=state,
    )


def _switch(parent: tk.Misc, text: str, variable: tk.Variable, command: Callable[[], None] | None = None) -> ctk.CTkSwitch:
    return ctk.CTkSwitch(
        parent,
        text=text,
        variable=variable,
        command=command,
        fg_color=BORDER,
        progress_color=ACCENT,
        button_color=TEXT,
        button_hover_color="#DDE4F0",
        text_color=TEXT,
        font=_font(13),
    )


def _checkbox(parent: tk.Misc, text: str, variable: tk.Variable, command: Callable[[], None] | None = None) -> ctk.CTkCheckBox:
    return ctk.CTkCheckBox(
        parent,
        text=text,
        variable=variable,
        command=command,
        width=22,
        height=22,
        corner_radius=6,
        border_width=2,
        border_color="#475569",
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color=TEXT,
        font=_font(13),
    )


def _entry(parent: tk.Misc, variable: tk.Variable, *, secret: bool = False) -> ctk.CTkEntry:
    return ctk.CTkEntry(
        parent,
        textvariable=variable,
        show="•" if secret else "",
        height=42,
        corner_radius=10,
        border_width=1,
        border_color=BORDER,
        fg_color=INPUT,
        text_color=TEXT,
        placeholder_text_color=SUBTLE,
        font=_font(13),
    )


def _slider(
    parent: tk.Misc,
    *,
    row: int,
    label: str,
    value_variable: tk.Variable,
    variable: tk.Variable,
    start: float,
    end: float,
    command: Callable[[str], None],
) -> ctk.CTkSlider:
    header = ctk.CTkFrame(parent, fg_color="transparent")
    header.grid(row=row, column=0, sticky="ew", pady=(13, 0))
    header.grid_columnconfigure(0, weight=1)
    _label(header, label).grid(row=0, column=0, sticky="w")
    ctk.CTkLabel(
        header,
        textvariable=value_variable,
        text_color=TEXT,
        fg_color=ACCENT_SOFT,
        corner_radius=8,
        width=64,
        height=27,
        font=_font(12, "bold"),
    ).grid(row=0, column=1, sticky="e")
    slider = ctk.CTkSlider(
        parent,
        from_=start,
        to=end,
        variable=variable,
        command=command,
        height=18,
        border_width=0,
        fg_color=INPUT,
        progress_color=ACCENT,
        button_color=TEXT,
        button_hover_color=ACCENT_HOVER,
    )
    slider.grid(row=row + 1, column=0, sticky="ew", pady=(8, 0))
    return slider


def build_main_layout(app: tk.Misc) -> None:
    ctk.set_appearance_mode("dark")
    app.configure(fg_color=BG)
    app.grid_columnconfigure(0, weight=1)
    app.grid_rowconfigure(1, weight=1)

    header = ctk.CTkFrame(app, fg_color=BG, corner_radius=0)
    header.grid(row=0, column=0, sticky="ew", padx=32, pady=(24, 14))
    header.grid_columnconfigure(0, weight=1)

    brand = ctk.CTkFrame(header, fg_color="transparent")
    brand.grid(row=0, column=0, sticky="w")
    mark = ctk.CTkLabel(
        brand,
        text="SC",
        width=48,
        height=48,
        corner_radius=14,
        fg_color=ACCENT,
        text_color="#FFFFFF",
        font=_font(16, "bold"),
    )
    mark.grid(row=0, column=0, rowspan=2, padx=(0, 14))
    _label(brand, "Speech Cutter", text_color=TEXT, font=_font(25, "bold")).grid(row=0, column=1, sticky="sw")
    _label(brand, "Szybki montaż mowy, napisów i obrazu").grid(row=1, column=1, sticky="nw", pady=(2, 0))

    badge = ctk.CTkLabel(
        header,
        text="  LOCAL + CLOUD  ",
        height=30,
        corner_radius=15,
        fg_color="#102D2E",
        text_color=SUCCESS,
        font=_font(11, "bold"),
    )
    badge.grid(row=0, column=1, sticky="e")

    content = ctk.CTkFrame(app, fg_color="transparent", height=520)
    content.grid(row=1, column=0, sticky="nsew", padx=32)
    content.grid_propagate(False)
    app.main_content = content
    content.grid_columnconfigure(0, weight=1)
    content.grid_rowconfigure(1, weight=1)

    pages: dict[str, ctk.CTkScrollableFrame] = {}
    page_names = ("Przygotowanie", "Obraz", "Dźwięk")

    def show_page(name: str) -> None:
        for page_name, page in pages.items():
            if page_name == name:
                page.grid()
            else:
                page.grid_remove()
        if hasattr(app, "footer_panel"):
            app.footer_panel.lift()

    app.section_picker = ctk.CTkSegmentedButton(
        content,
        values=list(page_names),
        command=show_page,
        height=42,
        corner_radius=11,
        fg_color=SURFACE,
        selected_color=ACCENT,
        selected_hover_color=ACCENT_HOVER,
        unselected_color=SURFACE,
        unselected_hover_color=CARD_HOVER,
        text_color=TEXT,
        font=_font(13, "bold"),
    )
    app.section_picker.grid(row=0, column=0, sticky="w", pady=(0, 14))

    for page_name in page_names:
        page = ctk.CTkScrollableFrame(
            content,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT,
        )
        page.grid(row=1, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=1, uniform="page")
        page.grid_columnconfigure(1, weight=1, uniform="page")
        pages[page_name] = page

    # Preparation page: source and transcript.
    setup_page = pages["Przygotowanie"]
    source_card, source_body = _card(
        setup_page,
        title="Wybierz materiał",
        subtitle="Dodaj jeden film albo włącz kolejkę, aby przetworzyć całą serię tymi samymi ustawieniami.",
    )
    source_card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
    source_body.grid_columnconfigure(0, weight=1)
    app.input_entry = _entry(source_body, app.input_var)
    app.input_entry.grid(row=0, column=0, sticky="ew")
    app.browse_button = _button(source_body, "Wybierz film", app._browse_input, primary=True, width=138)
    app.browse_button.grid(row=0, column=1, padx=(12, 0))
    app.queue_mode_check = _switch(source_body, "Tryb seryjny", app.queue_enabled_var, app._toggle_queue_mode)
    app.queue_mode_check.grid(row=0, column=2, padx=(18, 0))
    _label(source_body, textvariable=app.source_info_var, wraplength=980).grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(10, 0)
    )

    app.queue_frame = ctk.CTkFrame(source_body, fg_color=INPUT, corner_radius=12)
    app.queue_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(14, 0))
    app.queue_frame.grid_columnconfigure(0, weight=1)
    queue_header = ctk.CTkFrame(app.queue_frame, fg_color="transparent")
    queue_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
    queue_header.grid_columnconfigure(0, weight=1)
    _label(queue_header, "Kolejka", text_color=TEXT, font=_font(13, "bold")).grid(row=0, column=0, sticky="w")
    _label(queue_header, textvariable=app.queue_summary_var).grid(row=0, column=1, sticky="e")
    app.queue_listbox = tk.Listbox(
        app.queue_frame,
        height=4,
        activestyle="none",
        exportselection=False,
        relief="flat",
        bg=INPUT,
        fg=TEXT,
        highlightthickness=0,
        selectbackground=ACCENT_SOFT,
        selectforeground=TEXT,
        font=("Segoe UI", 11),
        borderwidth=0,
    )
    app.queue_listbox.grid(row=1, column=0, sticky="ew", padx=14)
    app.queue_listbox.bind("<<ListboxSelect>>", app._on_queue_selection)
    queue_actions = ctk.CTkFrame(app.queue_frame, fg_color="transparent")
    queue_actions.grid(row=2, column=0, sticky="w", padx=14, pady=(8, 12))
    app.remove_queue_button = _button(queue_actions, "Usuń wybrany", app._remove_selected_video, width=130)
    app.remove_queue_button.grid(row=0, column=0)
    app.clear_queue_button = _button(queue_actions, "Wyczyść", app._clear_queue, width=100)
    app.clear_queue_button.grid(row=0, column=1, padx=(8, 0))

    speech_card, speech_body = _card(
        setup_page,
        title="Cięcia mowy",
        subtitle="Naturalnie usuwa ciszę, zostawiając wygodny oddech przed i po wypowiedzi.",
    )
    speech_card.grid(row=1, column=0, sticky="nsew", padx=(0, 7), pady=(0, 14))
    speech_body.grid_columnconfigure(0, weight=1)
    app.padding_scale = _slider(
        speech_body,
        row=0,
        label="Margines wokół mowy",
        value_variable=app.padding_value_var,
        variable=app.padding_seconds_var,
        start=0.0,
        end=3.0,
        command=app._on_padding_scale,
    )
    _label(speech_body, "Najczęściej naturalnie brzmi 0,10–0,25 s.", text_color=SUBTLE).grid(
        row=2, column=0, sticky="w", pady=(14, 0)
    )

    captions_card, captions_body = _card(
        setup_page,
        title="Napisy i transkrypcja",
        subtitle="Lokalny Whisper albo szybka transkrypcja w chmurze Groq.",
    )
    captions_card.grid(row=1, column=1, sticky="nsew", padx=(7, 0), pady=(0, 14))
    captions_body.grid_columnconfigure(0, weight=1)
    _checkbox(captions_body, "Dodaj animowane napisy", app.captions_enabled_var).grid(row=0, column=0, sticky="w")
    _checkbox(captions_body, "Wycisz i ocenzuruj przekleństwa", app.profanity_filter_var).grid(
        row=1, column=0, sticky="w", pady=(10, 0)
    )
    provider_row = ctk.CTkFrame(captions_body, fg_color="transparent")
    provider_row.grid(row=2, column=0, sticky="ew", pady=(16, 0))
    provider_row.grid_columnconfigure(1, weight=1)
    _label(provider_row, "Silnik").grid(row=0, column=0, sticky="w")
    app.transcription_combobox = ctk.CTkOptionMenu(
        provider_row,
        variable=app.transcription_provider_var,
        values=("Local Whisper", "Groq API"),
        command=lambda _value: app._update_transcription_controls(),
        height=38,
        corner_radius=9,
        fg_color=INPUT,
        button_color=BORDER,
        button_hover_color=ACCENT,
        dropdown_fg_color=SURFACE,
        dropdown_hover_color=ACCENT_SOFT,
        text_color=TEXT,
        font=_font(13),
        dropdown_font=_font(13),
    )
    app.transcription_combobox.grid(row=0, column=1, sticky="ew", padx=(14, 0))
    app.groq_settings_frame = ctk.CTkFrame(captions_body, fg_color="transparent")
    app.groq_settings_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
    app.groq_settings_frame.grid_columnconfigure(0, weight=1)
    _label(app.groq_settings_frame, "Klucz Groq API").grid(row=0, column=0, sticky="w")
    _entry(app.groq_settings_frame, app.groq_api_key_var, secret=True).grid(row=1, column=0, sticky="ew", pady=(6, 0))
    _label(app.groq_settings_frame, "Model").grid(row=2, column=0, sticky="w", pady=(10, 0))
    ctk.CTkOptionMenu(
        app.groq_settings_frame,
        variable=app.groq_model_var,
        values=("whisper-large-v3-turbo", "whisper-large-v3"),
        height=38,
        corner_radius=9,
        fg_color=INPUT,
        button_color=BORDER,
        button_hover_color=ACCENT,
        dropdown_fg_color=SURFACE,
        dropdown_hover_color=ACCENT_SOFT,
        text_color=TEXT,
        font=_font(13),
        dropdown_font=_font(13),
    ).grid(row=3, column=0, sticky="ew", pady=(6, 0))
    app.polish_check = _checkbox(captions_body, "Wymuś rozpoznawanie polskiego", app.enhanced_polish_recognition_var)
    app.polish_check.grid(row=4, column=0, sticky="w", pady=(12, 0))
    _checkbox(captions_body, "Sprawdź logikę transkryptu przez Gemini", app.ai_correction_enabled_var,
              app._update_ai_controls).grid(row=5, column=0, sticky="w", pady=(10, 0))
    app.gemini_settings_frame = ctk.CTkFrame(captions_body, fg_color="transparent")
    app.gemini_settings_frame.grid(row=6, column=0, sticky="ew", pady=(12, 0))
    app.gemini_settings_frame.grid_columnconfigure(0, weight=1)
    _label(app.gemini_settings_frame, "Klucz Gemini API").grid(row=0, column=0, sticky="w")
    _entry(app.gemini_settings_frame, app.gemini_api_key_var, secret=True).grid(row=1, column=0, sticky="ew", pady=(6, 0))
    _label(app.gemini_settings_frame, "Klucz Groq fallback 1 (opcjonalny)").grid(
        row=2, column=0, sticky="w", pady=(12, 0)
    )
    _entry(app.gemini_settings_frame, app.groq_correction_api_key_var, secret=True).grid(
        row=3, column=0, sticky="ew", pady=(6, 0)
    )
    for key_number, key_var in enumerate(app.groq_correction_extra_api_key_vars, start=2):
        row = 2 + key_number * 2
        _label(app.gemini_settings_frame, f"Klucz Groq fallback {key_number} (opcjonalny)").grid(
            row=row, column=0, sticky="w", pady=(12, 0)
        )
        _entry(app.gemini_settings_frame, key_var, secret=True).grid(
            row=row + 1, column=0, sticky="ew", pady=(6, 0)
        )
    _label(
        app.gemini_settings_frame,
        "Automatycznie zastępują Gemini przy limicie API i rotują między kluczami. "
        "Są niezależne od klucza Groq do transkrypcji.",
        text_color=SUBTLE,
        font=_font(11),
        wraplength=540,
    ).grid(row=16, column=0, sticky="w", pady=(7, 0))
    _checkbox(captions_body, "Zapamiętaj klucze bezpiecznie", app.remember_api_keys_var).grid(
        row=7, column=0, sticky="w", pady=(12, 0)
    )

    # Video page.
    video_page = pages["Obraz"]
    crop_card, crop_body = _card(
        video_page,
        title="Dynamiczny kadr",
        subtitle="Automatycznie przybliża wskazany fragment obrazu w wybranych segmentach mowy.",
    )
    crop_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7), pady=(0, 14))
    crop_body.grid_columnconfigure(0, weight=1)
    _switch(crop_body, "Włącz dynamiczny kadr", app.crop_enabled_var, app._update_crop_state).grid(
        row=0, column=0, sticky="w"
    )
    crop_actions = ctk.CTkFrame(crop_body, fg_color="transparent")
    crop_actions.grid(row=1, column=0, sticky="w", pady=(14, 0))
    app.pick_crop_button = _button(crop_actions, "Wybierz kadr", app._choose_crop_visually, primary=True, width=125)
    app.pick_crop_button.grid(row=0, column=0)
    app.pick_queue_crops_button = _button(crop_actions, "Kadry dla kolejki", app._choose_crops_for_queue, width=145)
    app.pick_queue_crops_button.grid(row=0, column=1, padx=(8, 0))
    app.clear_crop_button = _button(crop_actions, "Wyczyść", app._clear_crop_selection, width=92)
    app.clear_crop_button.grid(row=0, column=2, padx=(8, 0))
    _label(crop_body, textvariable=app.crop_summary_var, wraplength=560).grid(
        row=2, column=0, sticky="w", pady=(12, 0)
    )
    app.crop_every_scale = _slider(
        crop_body, row=3, label="Co który segment mowy", value_variable=app.crop_every_value_var,
        variable=app.crop_every_var, start=1, end=12, command=app._on_crop_every_scale
    )
    app.crop_min_scale = _slider(
        crop_body, row=5, label="Pomiń segmenty krótsze niż", value_variable=app.crop_min_seconds_value_var,
        variable=app.crop_min_seconds_var, start=0.0, end=10.0, command=app._on_crop_min_scale
    )
    app.crop_zoom_scale = _slider(
        crop_body, row=7, label="Czas przybliżenia", value_variable=app.crop_zoom_seconds_value_var,
        variable=app.crop_zoom_seconds_var, start=0.0, end=10.0, command=app._on_crop_zoom_scale
    )

    motion_card, motion_body = _card(
        video_page,
        title="Ruch i dynamika",
        subtitle="Subtelny push-in dodaje energii dłuższym wypowiedziom bez gwałtownych skoków obrazu.",
    )
    motion_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0), pady=(0, 14))
    motion_body.grid_columnconfigure(0, weight=1)
    app.crop_middle_long_check = _checkbox(
        motion_body, "Przybliż środek długiego segmentu", app.crop_middle_long_var
    )
    app.crop_middle_long_check.grid(row=0, column=0, sticky="w")
    app.crop_middle_long_threshold_scale = _slider(
        motion_body, row=1, label="Długi segment od", value_variable=app.crop_middle_long_threshold_value_var,
        variable=app.crop_middle_long_threshold_var, start=4.0, end=30.0,
        command=app._on_crop_middle_long_threshold_scale
    )
    divider = ctk.CTkFrame(motion_body, height=1, fg_color=BORDER)
    divider.grid(row=3, column=0, sticky="ew", pady=(18, 15))
    _switch(motion_body, "Płynny push-in podczas mowy", app.push_in_enabled_var).grid(row=4, column=0, sticky="w")
    app.push_in_min_scale = _slider(
        motion_body, row=5, label="Włącz dla segmentów od", value_variable=app.push_in_min_seconds_value_var,
        variable=app.push_in_min_seconds_var, start=1.0, end=30.0, command=app._on_push_in_min_scale
    )
    app.push_in_zoom_scale = _slider(
        motion_body, row=7, label="Końcowe przybliżenie", value_variable=app.push_in_zoom_percent_value_var,
        variable=app.push_in_zoom_percent_var, start=2.0, end=25.0, command=app._on_push_in_zoom_scale
    )
    app._crop_widgets = [
        app.pick_crop_button, app.pick_queue_crops_button, app.clear_crop_button,
        app.crop_every_scale, app.crop_min_scale, app.crop_zoom_scale,
        app.crop_middle_long_check, app.crop_middle_long_threshold_scale,
    ]

    # Audio page.
    audio_page = pages["Dźwięk"]
    music_card, music_body = _card(
        audio_page,
        title="Muzyka w tle",
        subtitle="Podkład jest automatycznie zapętlany i miksowany pod oryginalną mowę.",
    )
    music_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7), pady=(0, 14))
    music_body.grid_columnconfigure(0, weight=1)
    _switch(music_body, "Dodaj Monkeys Spinning Monkeys", app.background_music_enabled_var).grid(
        row=0, column=0, sticky="w"
    )
    app.background_music_volume_scale = _slider(
        music_body, row=1, label="Głośność muzyki", value_variable=app.background_music_volume_value_var,
        variable=app.background_music_volume_percent_var, start=0.0, end=100.0,
        command=app._on_background_music_volume_scale
    )

    boost_card, boost_body = _card(
        audio_page,
        title="Wzmocnienie mowy",
        subtitle="Zwiększa głośność z limiterem chroniącym dźwięk przed przesterowaniem.",
    )
    boost_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0), pady=(0, 14))
    boost_body.grid_columnconfigure(0, weight=1)
    _switch(boost_body, "Wzmocnij głos", app.audio_boost_enabled_var).grid(row=0, column=0, sticky="w")
    app.audio_boost_scale = _slider(
        boost_body, row=1, label="Wzmocnienie", value_variable=app.audio_boost_gain_value_var,
        variable=app.audio_boost_gain_db_var, start=0.0, end=24.0, command=app._on_audio_boost_scale
    )

    # Persistent export bar.
    footer = ctk.CTkFrame(app, fg_color=SURFACE, corner_radius=18, border_width=1, border_color=BORDER)
    footer.grid(row=2, column=0, sticky="ew", padx=32, pady=(14, 24))
    app.footer_panel = footer
    footer.lift()
    footer.grid_columnconfigure(0, weight=1)
    status_area = ctk.CTkFrame(footer, fg_color="transparent")
    status_area.grid(row=0, column=0, sticky="ew", padx=(20, 12), pady=(16, 10))
    status_area.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(
        status_area,
        textvariable=app.status_var,
        text_color=TEXT,
        font=_font(13, "bold"),
        anchor="w",
    ).grid(row=0, column=0, sticky="ew")
    progress = ctk.CTkProgressBar(
        status_area,
        height=8,
        corner_radius=4,
        fg_color=INPUT,
        progress_color=ACCENT,
    )
    progress.grid(row=1, column=0, sticky="ew", pady=(9, 0))
    progress.set(max(0.0, min(float(app.progress_var.get()) / 100.0, 1.0)))
    app.progress_var.trace_add(
        "write",
        lambda *_args: progress.set(max(0.0, min(float(app.progress_var.get()) / 100.0, 1.0))),
    )
    app.open_folder_button = _button(footer, "Otwórz folder", app._open_output_folder, width=126, state="disabled")
    app.open_folder_button.grid(row=0, column=1, padx=(0, 8), pady=(16, 10))
    app.cancel_button = _button(footer, "Anuluj", app._cancel_processing, width=92, state="disabled")
    app.cancel_button.configure(fg_color="#3A1D2A", hover_color="#542336", text_color=DANGER)
    app.cancel_button.grid(row=0, column=2, padx=(0, 8), pady=(16, 10))
    app.start_button = _button(footer, "Utwórz film  →", app._start_processing, primary=True, width=158)
    app.start_button.grid(row=0, column=3, padx=(0, 18), pady=(16, 10))
    app.log_box = ctk.CTkTextbox(
        footer,
        height=78,
        corner_radius=10,
        border_width=0,
        fg_color=INPUT,
        text_color="#C8D4E7",
        scrollbar_button_color=BORDER,
        scrollbar_button_hover_color=ACCENT,
        font=("Cascadia Mono", 11),
        wrap="word",
    )
    app.log_box.grid(row=1, column=0, columnspan=4, sticky="ew", padx=18, pady=(0, 16))
    app.log_box.insert("end", "Gotowe. Wybierz film, aby rozpocząć.\n")
    app.log_box.configure(state="disabled")

    app.section_picker.set("Przygotowanie")
    show_page("Przygotowanie")
    app._update_transcription_controls()
    app._update_ai_controls()
