/**
 * File List Updater Extension
 * Automatically refreshes file lists when source_folder changes
 */

import { app } from "../../scripts/app.js";

console.log("[FileListUpdater] Loading File List Updater extension");

/**
 * Refresh a COMBO widget by fetching new data from API
 */
async function refreshFileList(fileWidget, apiRoute, sourceFolder) {
    console.log(`[FileListUpdater] === refreshFileList START ===`);
    console.log(`[FileListUpdater] fileWidget:`, fileWidget);
    console.log(`[FileListUpdater] apiRoute: ${apiRoute}`);
    console.log(`[FileListUpdater] sourceFolder: ${sourceFolder}`);
    console.log(`[FileListUpdater] Current widget value: ${fileWidget.value}`);
    console.log(`[FileListUpdater] Current widget options:`, fileWidget.options);

    try {
        const url = `${apiRoute}?source_folder=${sourceFolder}`;
        console.log(`[FileListUpdater] Fetching from: ${url}`);

        const response = await fetch(url);
        console.log(`[FileListUpdater] Response status: ${response.status}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const files = await response.json();
        console.log(`[FileListUpdater] Received ${files.length} files:`, files);

        // Update widget options
        console.log(`[FileListUpdater] Updating widget.options.values...`);
        fileWidget.options.values = files;
        console.log(`[FileListUpdater] Widget options after update:`, fileWidget.options);

        // Set value to first file or empty string
        if (files.length > 0) {
            console.log(`[FileListUpdater] Checking if current value "${fileWidget.value}" is in new list...`);
            // Keep current value if it exists in new list, otherwise select first
            if (!files.includes(fileWidget.value)) {
                console.log(`[FileListUpdater] Current value not found, selecting first file: ${files[0]}`);
                fileWidget.value = files[0];
            } else {
                console.log(`[FileListUpdater] Current value "${fileWidget.value}" still valid, keeping it`);
            }
        } else {
            console.log("[FileListUpdater] No files available, clearing widget value");
            fileWidget.value = "";
        }

        console.log(`[FileListUpdater] Final widget value: ${fileWidget.value}`);
        console.log(`[FileListUpdater] === refreshFileList END ===`);

    } catch (error) {
        console.error("[FileListUpdater] ERROR in refreshFileList:", error);
        console.error("[FileListUpdater] Error stack:", error.stack);
    }
}

/**
 * Setup dynamic file list for a node
 */
function setupDynamicFileList(node, nodeType) {
    console.log(`[FileListUpdater] === setupDynamicFileList START ===`);
    console.log(`[FileListUpdater] nodeType: ${nodeType}`);
    console.log(`[FileListUpdater] node:`, node);
    console.log(`[FileListUpdater] node.widgets:`, node.widgets);

    // Determine widget names and API route based on node type
    let fileWidgetName, apiRoute;

    if (nodeType === "LoadSMPL") {
        fileWidgetName = "npz_file";
        apiRoute = "/motioncapture/npz_files";
        console.log(`[FileListUpdater] Configured for LoadSMPL: fileWidgetName="${fileWidgetName}", apiRoute="${apiRoute}"`);
    } else if (nodeType === "LoadFBXCharacter") {
        fileWidgetName = "fbx_file";
        apiRoute = "/motioncapture/fbx_files";
        console.log(`[FileListUpdater] Configured for LoadFBXCharacter: fileWidgetName="${fileWidgetName}", apiRoute="${apiRoute}"`);
    } else {
        console.error("[FileListUpdater] Unknown node type:", nodeType);
        return;
    }

    // List all available widgets for debugging
    if (node.widgets) {
        console.log(`[FileListUpdater] Available widgets (${node.widgets.length}):`);
        node.widgets.forEach((w, i) => {
            console.log(`[FileListUpdater]   [${i}] name="${w.name}", type="${w.type}", value="${w.value}"`);
        });
    } else {
        console.error("[FileListUpdater] No widgets found on node!");
    }

    // Find widgets
    console.log(`[FileListUpdater] Looking for source_folder widget...`);
    const sourceFolderWidget = node.widgets?.find(w => w.name === "source_folder");

    console.log(`[FileListUpdater] Looking for ${fileWidgetName} widget...`);
    const fileWidget = node.widgets?.find(w => w.name === fileWidgetName);

    if (!sourceFolderWidget) {
        console.error(`[FileListUpdater] Could not find source_folder widget in ${nodeType}`);
        return;
    } else {
        console.log(`[FileListUpdater] Found source_folder widget:`, sourceFolderWidget);
        console.log(`[FileListUpdater] source_folder current value: "${sourceFolderWidget.value}"`);
        console.log(`[FileListUpdater] source_folder options:`, sourceFolderWidget.options);
    }

    if (!fileWidget) {
        console.error(`[FileListUpdater] Could not find ${fileWidgetName} widget in ${nodeType}`);
        return;
    } else {
        console.log(`[FileListUpdater] Found ${fileWidgetName} widget:`, fileWidget);
        console.log(`[FileListUpdater] ${fileWidgetName} current value: "${fileWidget.value}"`);
        console.log(`[FileListUpdater] ${fileWidgetName} options:`, fileWidget.options);
    }

    console.log(`[FileListUpdater] Setting up ${nodeType} file list updater`);

    // Store original callback if it exists
    const originalCallback = sourceFolderWidget.callback;
    console.log(`[FileListUpdater] Original callback exists: ${!!originalCallback}`);

    // Override callback to refresh file list when source_folder changes
    sourceFolderWidget.callback = function(value) {
        console.log(`[FileListUpdater] === source_folder CALLBACK TRIGGERED ===`);
        console.log(`[FileListUpdater] source_folder changed to: ${value}`);

        // Call original callback if it exists
        if (originalCallback) {
            console.log(`[FileListUpdater] Calling original callback...`);
            originalCallback.apply(this, arguments);
        }

        // Refresh the file list
        console.log(`[FileListUpdater] Refreshing file list...`);
        refreshFileList(fileWidget, apiRoute, value);
    };

    console.log(`[FileListUpdater] Callback override complete`);

    // Initial load - fetch files for current source_folder value
    console.log(`[FileListUpdater] Performing initial load for ${nodeType}`);
    console.log(`[FileListUpdater] Initial source_folder value: "${sourceFolderWidget.value}"`);
    refreshFileList(fileWidget, apiRoute, sourceFolderWidget.value);

    console.log(`[FileListUpdater] === setupDynamicFileList END ===`);
}

// Register extension
console.log("[FileListUpdater] Registering extension with app...");

app.registerExtension({
    name: "Comfy.MotionCapture.FileListUpdater",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log(`[FileListUpdater] beforeRegisterNodeDef called for: ${nodeData.name}`);

        // Handle both LoadSMPL and LoadFBXCharacter nodes
        if (nodeData.name === "LoadSMPL" || nodeData.name === "LoadFBXCharacter") {
            console.log(`[FileListUpdater] ✓ Matched node type: ${nodeData.name}`);
            console.log(`[FileListUpdater] Registering extension for ${nodeData.name}`);
            console.log(`[FileListUpdater] nodeType:`, nodeType);
            console.log(`[FileListUpdater] nodeData:`, nodeData);

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            console.log(`[FileListUpdater] Original onNodeCreated exists: ${!!onNodeCreated}`);

            nodeType.prototype.onNodeCreated = function() {
                console.log(`[FileListUpdater] === onNodeCreated CALLED for ${nodeData.name} ===`);
                console.log(`[FileListUpdater] this:`, this);

                const result = onNodeCreated?.apply(this, arguments);
                console.log(`[FileListUpdater] Original onNodeCreated returned:`, result);

                // Setup dynamic file list updating
                console.log(`[FileListUpdater] Calling setupDynamicFileList...`);
                setupDynamicFileList(this, nodeData.name);

                console.log(`[FileListUpdater] === onNodeCreated COMPLETE ===`);
                return result;
            };

            console.log(`[FileListUpdater] onNodeCreated override complete for ${nodeData.name}`);
        } else {
            console.log(`[FileListUpdater] ✗ Skipping node type: ${nodeData.name}`);
        }
    }
});

console.log("[FileListUpdater] Extension registered successfully");
