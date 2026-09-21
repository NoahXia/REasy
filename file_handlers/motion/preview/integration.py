from __future__ import annotations

from .support_registry import entity_motion_support_for_game


def _is_wots_game(value: str) -> bool:
    normalized = "".join(
        character
        for character in str(value or "")
        if character.isalnum()
    ).casefold()
    return normalized in {"onimushawots", "oniwots", "wots"}


def _resource_context_available(handler) -> bool:
    context = getattr(handler, "resource_context", None)
    return context is not None and any((
        getattr(context, "project_dir", ""),
        getattr(context, "unpacked_dir", ""),
        getattr(context, "pak_cached_reader", None),
    ))


def create_pfb_motion_preview(handler):
    if not getattr(getattr(handler, "rsz_file", None), "is_pfb", False):
        return None
    game_version = getattr(handler, "game_version", "")
    support = entity_motion_support_for_game(
        game_version
    )
    is_wots = _is_wots_game(game_version)
    if support is None and not is_wots:
        return None
    if not _resource_context_available(handler):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QLabel

        notice = QLabel(
            handler.tr(
                "PFB 3D preview is only available in project mode. "
                "Please open or create a project."
            )
        )
        notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        notice.setWordWrap(True)
        return notice

    if is_wots:
        from file_handlers.rsz.wots_pfb_preview import WotsPfbPreviewWidget

        return WotsPfbPreviewWidget(handler)

    from .entity_widget import PfbMotionPreviewWidget

    return PfbMotionPreviewWidget(handler, support=support)
