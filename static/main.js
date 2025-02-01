import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, MToonMaterialLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { MToonNodeMaterial } from '@pixiv/three-vrm/nodes';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js'

const MODEL_PATH = 'http://127.0.0.1:5500/models/visual/vrm_models/Diamond.vrm'

const socket = new WebSocket("ws://127.0.0.1:8765");


class ModelViewer {
    constructor() {
        this.clock = new THREE.Clock(); // Создаем таймер
        this.initScene();
        this.initControls();
        this.initAudio();
        this.setupEventListeners();
        this.preloadModel();
        this.isDragging = false;
        this.dragStartMouse = new THREE.Vector2();
        this.dragStartModelPosition = new THREE.Vector3();
        this.setupDragHandlers();
        this.draggedObject = null;

        // Добавляем mixer для анимаций
        this.mixer = null;
        this.animationActions = [];
        this.activeAction = null;
        this.lastAction = null;
        this.modelReady = false;

        // Загрузчики
        this.gltfLoader = new GLTFLoader();
        this.fbxLoader = new FBXLoader();

        // Группа для модели
        this.modelGroup = new THREE.Group();
        this.scene.add(this.modelGroup);

        // Хранилище предзагруженных моделей
        this.preloadedAssets = new Map(); // Инициализируем Map

        this.vrm = null; // Храним VRM-модель

        // Подключение WebSocket
        this.socket = new WebSocket("ws://127.0.0.1:8765");
        this.socket.onmessage = this.handleWebSocketMessage.bind(this);
    }

    async initScene() {
        // Создаем WebGL-совместимый canvas
        const canvas = document.createElement('canvas');
        document.body.appendChild(canvas);

        // Камера
        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 1.6, 3);

        // Сцена
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xeeeeee);

        // Рендерер
        this.renderer = new THREE.WebGPURenderer({
            canvas: document.querySelector('canvas'),
            antialias: true,
            alpha: true,
        });

        this.model = MODEL_PATH;
        console.log("Model init from: ", this.model);

        try {
            await this.renderer.init(); // Асинхронная инициализация
            console.log("WebGPURenderer initialized successfully");

            // Настройка размеров и пиксельного соотношения
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.setPixelRatio(window.devicePixelRatio);

            // Включение теней (если необходимо)
            this.renderer.shadowMap.enabled = true;

            // Запуск цикла рендеринга
            this.animate();
        } catch (error) {
            console.error("Failed to initialize WebGPURenderer:", error);
        }

        // Освещение
        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        this.directionalLight.position.set(5, 5, 5);
        this.directionalLight.castShadow = true;
        this.scene.add(this.ambientLight, this.directionalLight);

        // LOD (Level of Detail)
        this.lod = new THREE.LOD();
        this.scene.add(this.lod);
    }

    handleWebSocketMessage(event) {
        const rawData = event.data;
        console.log("Получены данные:", rawData);

        if (rawData.includes("/VMC/Ext/Bone/Pos")) {
            const boneData = this.parseBonePosition(rawData);
            this.updateModel(boneData);
        } else if (rawData.includes("/VMC/Ext/Blend/Apply")) {
            const blendshapeData = this.parseBlendshapes(rawData);
            this.updateBlendshapes(blendshapeData);
        }
    }

    parseBonePosition(rawData) {
        const parts = rawData.split(",");
        const boneName = parts[2];
        const position = {
            x: parseFloat(parts[3]),
            y: parseFloat(parts[4]),
            z: parseFloat(parts[5])
        };
        const rotation = {
            x: parseFloat(parts[6]),
            y: parseFloat(parts[7]),
            z: parseFloat(parts[8]),
            w: parseFloat(parts[9])
        };
        return { boneName, position, rotation };
    }

    updateModel(boneData) {
        if (!this.vrm) {
            console.warn("VRM model is not loaded.");
            return;
        }

        const bone = this.vrm.humanoid.getBoneNode(boneData.boneName);
        if (bone) {
            bone.position.set(boneData.position.x, boneData.position.y, boneData.position.z);
            bone.quaternion.set(
                boneData.rotation.x,
                boneData.rotation.y,
                boneData.rotation.z,
                boneData.rotation.w
            );
        } else {
            console.warn(`Bone "${boneData.boneName}" not found in VRM model.`);
        }
    }

    parseBlendshapes(rawData) {
        const parts = rawData.split(",");
        const blendshapes = {};
        for (let i = 2; i < parts.length; i += 2) {
            const name = parts[i];
            const value = parseFloat(parts[i + 1]);
            blendshapes[name] = value;
        }
        return blendshapes;
    }

    updateBlendshapes(blendshapeData) {
        if (!this.vrm || !this.vrm.expressionManager) {
            console.warn("VRM expression manager is not available.");
            return;
        }

        for (const [name, value] of Object.entries(blendshapeData)) {
            this.vrm.expressionManager.setValue(name, value);
        }
        this.vrm.expressionManager.update();
    }

    async preloadModel() {
        if (!this.model) {
            console.error("Model is None?");
            return;
        }

        try {
            console.log(`Loading model from: ${this.model}`);
            const response = await fetch(this.model);
            if (!response.ok) {
                throw new Error(`Failed to load model: ${this.model}`);
            }
            const arrayBuffer = await response.arrayBuffer();

            // gltf and vrm
            const loader = new GLTFLoader();
            loader.crossOrigin = 'anonymous';
            loader.register((parser) => {
                const mtoonMaterialPlugin = new MToonMaterialLoaderPlugin(parser, {
                    materialType: MToonNodeMaterial,
                });
                return new VRMLoaderPlugin(parser, {
                    mtoonMaterialPlugin,
                });
            });

            loader.parse(arrayBuffer, '', (gltf) => {
                this.preloadedAssets.set(this.model, gltf);
                console.log('VRM model loaded:', gltf.userData.vrm);
                this.loadModel(gltf); // Загружаем модель
            }, (error) => {
                console.error('Error parsing model:', error);
            });
        } catch (error) {
            console.error('Preloading failed:', error);
        }
    }

    loadModel(gltf) {
        try {
            // Проверяем, существует ли modelGroup
            if (!this.modelGroup) {
                console.error("modelGroup is not initialized.");
                return;
            }

            // Очистка предыдущей модели
            this.modelGroup.clear();
            if (this.mixer) {
                this.mixer.stopAllAction();
                this.mixer = null;
            }

            const vrm = gltf.userData.vrm; // Получаем VRM-модель
            if (vrm) {
                console.log('VRM model loaded:', vrm);
                this.modelGroup.add(vrm.scene); // Добавляем модель в группу

                // Настройка морфинга (например, улыбка)
                const expressionManager = vrm.expressionManager;
                if (expressionManager) {
                    expressionManager.setValue('sad', 1.0); // Устанавливаем вес эмоции
                    expressionManager.update(); // Применяем изменения
                }

                // Настройка анимаций
                this.mixer = new THREE.AnimationMixer(vrm.scene);

            } else {
                console.warn('Loaded model is not a VRM.');
            }
        } catch (error) {
            console.error('Model loading failed:', error);
        }
    }

    setActiveAction(toAction) {
        if (toAction !== this.activeAction) {
            if (this.activeAction) {
                this.activeAction.fadeOut(0.5);
            }

            this.activeAction = toAction;
            this.activeAction.reset();
            this.activeAction.fadeIn(0.5);
            this.activeAction.play();
        }
    }

    toggleAnimation() {
        if (this.mixer) {
            this.mixer.timeScale = this.mixer.timeScale === 0 ? 1 : 0;
        }
    }

    async animate() {
        if (!this.camera || !this.renderer || !this.scene) return;

        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();
        if (this.mixer) this.mixer.update(delta);
        if (this.controls) this.controls.update();

        await this.renderer.renderAsync(this.scene, this.camera);
    }

    initControls() {
        if (!this.camera || !this.renderer) {
            console.error("Camera or renderer is not initialized");
            return;
        }

        // Управление камерой
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 1;
        this.controls.minDistance = 0.01;
        this.controls.maxDistance = 7;
    }

    setupDragHandlers() {
        window.addEventListener('mousedown', (event) => this.onMouseDown(event));
        window.addEventListener('mousemove', (event) => this.onMouseMove(event));
        window.addEventListener('mouseup', () => this.onMouseUp());
    }
    
    onMouseDown(event) {
        const mouse = new THREE.Vector2();
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);

        const intersects = raycaster.intersectObjects(this.modelGroup.children, true);
        if (intersects.length > 0) {
            const clickedObject = intersects[0].object;

            // Отключаем OrbitControls
            this.controls.enabled = false;

            // Начинаем перетаскивание
            this.isDragging = true;
            this.dragStartMouse.set(event.clientX, event.clientY);
            this.dragStartModelPosition.copy(clickedObject.position);

            // Сохраняем ссылку на перемещаемый объект
            this.draggedObject = clickedObject;
        } else {
            console.warn("No object was clicked.");
        }
    }
    
    onMouseUp() {
        this.isDragging = false;
        this.draggedObject = null;
    
        // Включаем OrbitControls
        this.controls.enabled = true;
    }
    
    onMouseMove(event) {
        if (!this.isDragging || !this.draggedObject) return;

        const scaleFactor = 0.001;
        const deltaX = (event.clientX - this.dragStartMouse.x) * scaleFactor;
        const deltaY = (event.clientY - this.dragStartMouse.y) * scaleFactor;

        if (this.draggedObject && this.draggedObject.position) {
            this.draggedObject.position.x = this.dragStartModelPosition.x + deltaX;
            this.draggedObject.position.y = this.dragStartModelPosition.y - deltaY;
        }
    }
    
    onMouseUp() {
        this.isDragging = false;
        this.draggedObject = null;

        // Включаем OrbitControls
        this.controls.enabled = true;
    }

    setupEventListeners() {
        // Обработка событий
        window.addEventListener('resize', () => this.onWindowResize());
        window.addEventListener('click', e => this.handleClick(e));
        document.addEventListener('keydown', e => {
            if (e.key === ' ') this.toggleAnimation();
        });
        document.addEventListener('click', () => {
            if (THREE.AudioContext) {
                THREE.AudioContext.getContext().resume();
            }
        }, { once: true });
    }

    updateFaceExpression(vrm, expressionName, weight) {
        const expressionManager = vrm.expressionManager;
        if (expressionManager && expressionManager.setValue) {
            expressionManager.setValue(expressionName, weight);
            expressionManager.update();
        }
    }

    handleClick(event) {
        // Raycaster
        const mouse = new THREE.Vector2();

        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        const raycaster = new THREE.Raycaster();
        
        raycaster.setFromCamera(mouse, this.camera);

        const intersects = raycaster.intersectObjects(this.modelGroup.children, true);

        if (intersects.length > 0) {
            const object = intersects[0].object;
            this.highlightObject(object);
            this.playSound(this.clickSound);
        }
    }

    highlightObject(object) {
        if (object.material) {
            object.material.color.setHex(Math.random() * 0xffffff);
        }
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    initAudio() {
        // Аудио система
        this.audioListener = new THREE.AudioListener();
        this.camera.add(this.audioListener);

        // Фоновая музыка
        // this.backgroundMusic = new THREE.Audio(this.audioListener);
        // new THREE.AudioLoader().load('http://127.0.0.1:5500/tools/sounds/forest-sound.mp3', buffer => {
        //     this.backgroundMusic.setBuffer(buffer);
        //     this.backgroundMusic.setLoop(true);
        //     this.backgroundMusic.setVolume(0.3);
        //     this.backgroundMusic.play();
        // });

        // Звуки кликов
        this.clickSound = new THREE.Audio(this.audioListener);
        new THREE.AudioLoader().load('http://127.0.0.1:5500/tools/sounds/clicking.mp3', buffer => {
            this.clickSound.setBuffer(buffer);
            this.clickSound.setVolume(0.5);
        });
    }

    playSound(sound) {
        if (sound.isPlaying) sound.stop();
        sound.play();
    }
}

// Инициализация приложения
const viewer = new ModelViewer();
viewer.animate();