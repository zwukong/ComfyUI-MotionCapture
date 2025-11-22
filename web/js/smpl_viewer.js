/**
 * SMPL 3D Viewer - Interactive Three.js viewer for SMPL motion capture sequences
 * Uses locally bundled Three.js (no CDN dependencies)
 */

import * as THREE from '../lib/three.min.js';
import { OrbitControls } from '../lib/OrbitControls.js';

class SMPLViewer {
    constructor(container, data) {
        this.container = container;
        this.data = data;
        this.currentFrame = 0;
        this.isPlaying = false;
        this.animationId = null;

        // Create viewer elements
        this.createViewer();
        this.setupScene();
        this.createMesh();
        this.setupControls();
        this.animate();
    }

    createViewer() {
        // Create canvas and controls container
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'smpl-viewer-canvas';

        this.controlsContainer = document.createElement('div');
        this.controlsContainer.className = 'smpl-viewer-controls';

        // Play/Pause button
        this.playButton = document.createElement('button');
        this.playButton.className = 'smpl-control-button';
        this.playButton.innerHTML = '▶';
        this.playButton.onclick = () => this.togglePlayback();

        // Frame slider
        this.frameSlider = document.createElement('input');
        this.frameSlider.type = 'range';
        this.frameSlider.min = 0;
        this.frameSlider.max = this.data.frames - 1;
        this.frameSlider.value = 0;
        this.frameSlider.className = 'smpl-frame-slider';
        this.frameSlider.oninput = (e) => this.setFrame(parseInt(e.target.value));

        // Frame counter
        this.frameCounter = document.createElement('span');
        this.frameCounter.className = 'smpl-frame-counter';
        this.updateFrameCounter();

        // Assemble controls
        this.controlsContainer.appendChild(this.playButton);
        this.controlsContainer.appendChild(this.frameSlider);
        this.controlsContainer.appendChild(this.frameCounter);

        // Add to container
        this.container.appendChild(this.canvas);
        this.container.appendChild(this.controlsContainer);
    }

    setupScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);

        // Camera
        const width = this.container.clientWidth;
        const height = Math.min(width * 0.75, 600); // 4:3 aspect ratio, max 600px
        this.canvas.width = width;
        this.canvas.height = height;

        this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
        this.camera.position.set(0, 1, 3);
        this.camera.lookAt(0, 1, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 2, 1);
        this.scene.add(directionalLight);

        const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
        backLight.position.set(-1, 0, -1);
        this.scene.add(backLight);

        // Grid
        const gridHelper = new THREE.GridHelper(4, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);

        // Orbit controls
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.target.set(0, 1, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.update();
    }

    createMesh() {
        // Create BufferGeometry
        this.geometry = new THREE.BufferGeometry();

        // Set initial vertices (frame 0)
        const vertices = new Float32Array(this.data.vertices[0].flat());
        this.geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

        // Set faces (indices)
        const indices = new Uint32Array(this.data.faces.flat());
        this.geometry.setIndex(new THREE.BufferAttribute(indices, 1));

        // Compute normals for lighting
        this.geometry.computeVertexNormals();

        // Material
        const color = new THREE.Color(this.data.mesh_color);
        this.material = new THREE.MeshPhongMaterial({
            color: color,
            side: THREE.DoubleSide,
            flatShading: false
        });

        // Create mesh
        this.mesh = new THREE.Mesh(this.geometry, this.material);
        this.scene.add(this.mesh);
    }

    setupControls() {
        // Handle window resize
        window.addEventListener('resize', () => this.onResize());
    }

    onResize() {
        const width = this.container.clientWidth;
        const height = Math.min(width * 0.75, 600);

        this.canvas.width = width;
        this.canvas.height = height;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    setFrame(frameIndex) {
        this.currentFrame = Math.max(0, Math.min(frameIndex, this.data.frames - 1));

        // Update mesh vertices
        const vertices = new Float32Array(this.data.vertices[this.currentFrame].flat());
        this.geometry.attributes.position.array.set(vertices);
        this.geometry.attributes.position.needsUpdate = true;
        this.geometry.computeVertexNormals();

        // Update UI
        this.frameSlider.value = this.currentFrame;
        this.updateFrameCounter();
    }

    togglePlayback() {
        this.isPlaying = !this.isPlaying;
        this.playButton.innerHTML = this.isPlaying ? '⏸' : '▶';

        if (this.isPlaying) {
            this.lastFrameTime = performance.now();
        }
    }

    updateFrameCounter() {
        this.frameCounter.textContent = `${this.currentFrame + 1} / ${this.data.frames}`;
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());

        // Update playback
        if (this.isPlaying) {
            const now = performance.now();
            const deltaTime = now - this.lastFrameTime;
            const frameDuration = 1000 / this.data.fps; // ms per frame

            if (deltaTime >= frameDuration) {
                let nextFrame = this.currentFrame + 1;
                if (nextFrame >= this.data.frames) {
                    nextFrame = 0; // Loop
                }
                this.setFrame(nextFrame);
                this.lastFrameTime = now - (deltaTime % frameDuration);
            }
        }

        // Update controls and render
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this.geometry.dispose();
        this.material.dispose();
        this.renderer.dispose();
    }
}

// ComfyUI integration
export function createSMPLViewer(node, widgetValue) {
    const data = widgetValue[0];

    // Create container
    const container = document.createElement('div');
    container.className = 'smpl-viewer-container';
    container.style.width = '100%';
    container.style.marginTop = '10px';

    // Create viewer
    const viewer = new SMPLViewer(container, data);

    // Add to node widget
    node.addDOMWidget('smpl_viewer', 'SMPL_VIEWER', container, {
        serialize: false,
        hideOnZoom: false
    });

    // Cleanup on node removal
    node.onRemoved = () => {
        if (viewer) {
            viewer.dispose();
        }
    };

    return container;
}

// Auto-register with ComfyUI
if (typeof window.comfyAPI !== 'undefined') {
    window.comfyAPI.registerExtension({
        name: 'SMPL.Viewer',
        async nodeCreated(node) {
            if (node.comfyClass === 'SMPLViewer') {
                const onExecuted = node.onExecuted;
                node.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);
                    if (message?.smpl_viewer) {
                        createSMPLViewer(node, message.smpl_viewer);
                    }
                };
            }
        }
    });
}
