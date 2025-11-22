"""
FBX Animation Viewer Node - Interactive animation playback for animated FBX files
"""

from typing import Tuple
from hmr4d.utils.pylogger import Log


class FBXAnimationViewer:
    """
    Display an interactive animation viewer for animated FBX files.
    Shows skeletal animation playback with play/pause controls, timeline scrubber,
    and adjustable playback speed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_path": ("STRING", {
                    "forceInput": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "view_animation"
    OUTPUT_NODE = True
    CATEGORY = "MotionCapture"

    def view_animation(self, fbx_path: str) -> Tuple[str,]:
        """
        Display animated FBX playback in ComfyUI UI.

        Args:
            fbx_path: Absolute path to animated FBX file

        Returns:
            Tuple with fbx_path (passthrough)
        """
        try:
            Log.info(f"[FBXAnimationViewer] Displaying animation for: {fbx_path}")

            # The actual animation viewer is handled by the web extension
            # This node just passes through the path to the frontend
            return (fbx_path,)

        except Exception as e:
            error_msg = f"FBXAnimationViewer failed: {str(e)}"
            Log.error(error_msg)
            import traceback
            traceback.print_exc()
            return ("",)


NODE_CLASS_MAPPINGS = {
    "FBXAnimationViewer": FBXAnimationViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FBXAnimationViewer": "FBX Animation Viewer",
}
