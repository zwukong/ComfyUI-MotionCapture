/**
 * File List Updater Extension
 * Automatically refreshes file lists when source_folder changes
 */

import { app } from "../../scripts/app.js";

/**
 * Refresh a COMBO widget by fetching new data from API
 */
async function refreshFileList(fileWidget, apiRoute, sourceFolder) {
    try {
        const url = `${apiRoute}?source_folder=${sourceFolder}`;
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const files = await response.json();

        // Update widget options
        fileWidget.options.values = files;

        // Set value to first file or empty string
        if (files.length > 0) {
            // Keep current value if it exists in new list, otherwise select first
            if (!files.includes(fileWidget.value)) {
                fileWidget.value = files[0];
            }
        } else {
            fileWidget.value = "";
        }

    } catch (error) {
        console.error("[FileListUpdater] ERROR in refreshFileList:", error);
    }
}

/**
 * Setup dynamic file list for a node
 */
function setupDynamicFileList(node, nodeType) {
    // Determine widget names and API route based on node type
    let fileWidgetName, apiRoute;

    if (nodeType === "LoadSMPL") {
        fileWidgetName = "npz_file";
        apiRoute = "/motioncapture/npz_files";
    } else if (nodeType === "LoadFBXCharacter") {
        fileWidgetName = "fbx_file";
        apiRoute = "/motioncapture/fbx_files";
    } else {
        console.error("[FileListUpdater] Unknown node type:", nodeType);
        return;
    }

    // Find widgets
    const sourceFolderWidget = node.widgets?.find(w => w.name === "source_folder");
    const fileWidget = node.widgets?.find(w => w.name === fileWidgetName);

    if (!sourceFolderWidget) {
        console.error(`[FileListUpdater] Could not find source_folder widget in ${nodeType}`);
        return;
    }

    if (!fileWidget) {
        console.error(`[FileListUpdater] Could not find ${fileWidgetName} widget in ${nodeType}`);
        return;
    }

    // Store original callback if it exists
    const originalCallback = sourceFolderWidget.callback;

    // Override callback to refresh file list when source_folder changes
    sourceFolderWidget.callback = function(value) {
        // Call original callback if it exists
        if (originalCallback) {
            originalCallback.apply(this, arguments);
        }

        // Refresh the file list
        refreshFileList(fileWidget, apiRoute, value);
    };

    // Initial load - fetch files for current source_folder value
    setTimeout(() => {
        if (fileWidget && sourceFolderWidget) {
            refreshFileList(fileWidget, apiRoute, sourceFolderWidget.value);
        }
    }, 10); // 10ms delay to allow widgets to fully initialize
}

app.registerExtension({
    name: "Comfy.MotionCapture.FileListUpdater",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Handle both LoadSMPL and LoadFBXCharacter nodes
        if (nodeData.name === "LoadSMPL" || nodeData.name === "LoadFBXCharacter") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Setup dynamic file list updating
                setupDynamicFileList(this, nodeData.name);

                return result;
            };
        }
    }
});
