import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, MToonMaterialLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { MToonNodeMaterial } from '@pixiv/three-vrm/nodes';


class ModelViewer {
    constructor() {
        this.initScene();
        this.initControls();
        this.initAudio();
        this.setupEventListeners();
        this.preloadModels();
        this.isDragging = false;
        this.dragStartMouse = new THREE.Vector2();
        this.dragStartModelPosition = new THREE.Vector3();
        this.setupDragHandlers();
        this.draggedObject = null;
        this.setupMouseWheelZoom();
    }

    initScene() {
        // Сцена
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xeeeeee);

        // Камера
        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 1.6, 3);

        // Рендерер
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        document.body.appendChild(this.renderer.domElement);

        // Освещение
        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        this.directionalLight.position.set(5, 5, 5);
        this.directionalLight.castShadow = true;
        this.scene.add(this.ambientLight, this.directionalLight);

        // Загрузчик и таймер
        this.loader = new GLTFLoader();
        this.clock = new THREE.Clock();
        this.mixer = null;

        // Модели
        this.models = [
            'http://127.0.0.1:5500/models/visual/vrm_models/Ki.vrm'
        ]

        this.preloadedAssets = new Map();
        this.currentModelIndex = 0;
        this.modelGroup = new THREE.Group();
        this.scene.add(this.modelGroup);

        // LOD (Level of Detail)
        this.lod = new THREE.LOD();
        this.scene.add(this.lod);
    }

    async preloadModels() {
        try {
            await Promise.all(this.models.map(async (url, index) => {
                console.log(`Loading model from: ${url}`);
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`Failed to load model: ${url}`);
                }
                const arrayBuffer = await response.arrayBuffer();

                const loader = new THREE.GLTFLoader();
                loader.register((parser) => new VRMLoaderPlugin(parser)); // Добавляем поддержку VRM

                loader.parse(arrayBuffer, '', (gltf) => {
                    this.preloadedAssets.set(url, gltf);
                    console.log('VRM model loaded:', gltf.userData.vrm);
                    if (index === 0) {
                        this.loadModel(gltf); // Загружаем первую модель сразу после загрузки
                    }
                }, (error) => {
                    console.error('Error parsing model:', error);
                });
            }));
        } catch (error) {
            console.error('Preloading failed:', error);
        }
    }

    loadModel(gltf) {
        try {
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
                console.log(vrm);

                // Настройка морфинга (например, улыбка)
                const expressionManager = vrm.expressionManager;
                if (expressionManager) {
                    expressionManager.setValue('happy', 1.0); // Устанавливаем вес эмоции
                    expressionManager.update(); // Применяем изменения
                }

                // Настройка анимаций
                if (gltf.animations.length > 0) {
                    this.mixer = new THREE.AnimationMixer(vrm.scene);
                    this.mixer.clipAction(gltf.animations[0]).play();
                }
            } else {
                console.warn('Loaded model is not a VRM.');
            }
        } catch (error) {
            console.error('Model loading failed:', error);
        }
    }

    initControls() {
        // Управление камерой
        // this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 1;
        this.controls.minDistance = 1;
        this.controls.maxDistance = 10;
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

    setupMouseWheelZoom() {
        window.addEventListener('wheel', (event) => this.onMouseWheel(event), { passive: false });
    }
    
    onMouseWheel(event) {
        const mouse = new THREE.Vector2();
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);

        const intersects = raycaster.intersectObjects(this.scene.children, true);
        if (intersects.length > 0) {
            const targetPoint = intersects[0].point;

            const zoomFactor = event.deltaY > 0 ? 1.1 : 0.9;
            const direction = new THREE.Vector3().subVectors(targetPoint, this.camera.position).normalize();
            const distance = this.camera.position.distanceTo(targetPoint);

            this.camera.position.addScaledVector(direction, (zoomFactor - 1) * distance * 0.5);
            this.controls.target.copy(targetPoint);
            this.controls.update();
        }

        event.preventDefault();
    }

    initAudio() {
        // Аудио система
        this.audioListener = new THREE.AudioListener();
        this.camera.add(this.audioListener);

        // Фоновая музыка
        this.backgroundMusic = new THREE.Audio(this.audioListener);
        new THREE.AudioLoader().load('http://127.0.0.1:5500/tools/sounds/forest-sound.mp3', buffer => {
            this.backgroundMusic.setBuffer(buffer);
            this.backgroundMusic.setLoop(true);
            this.backgroundMusic.setVolume(0.3);
            this.backgroundMusic.play();
        });

        // Звуки кликов
        this.clickSound = new THREE.Audio(this.audioListener);
        new THREE.AudioLoader().load('http://127.0.0.1:5500/tools/sounds/clicking.mp3', buffer => {
            this.clickSound.setBuffer(buffer);
            this.clickSound.setVolume(0.5);
        });
    }

    setupEventListeners() {
        // Обработка событий
        window.addEventListener('resize', () => this.onWindowResize());
        window.addEventListener('click', e => this.handleClick(e));
        document.addEventListener('keydown', e => {
            if (e.key === 'm') this.nextModel();
            if (e.key === ' ') this.toggleAnimation();
        });
    }

    nextModel() {
        this.currentModelIndex = (this.currentModelIndex + 1) % this.models.length;
        this.loadModel();
        this.playSound(this.clickSound);
    }

    toggleAnimation() {
        if (this.mixer) {
            this.mixer.timeScale = this.mixer.timeScale === 0 ? 1 : 0;
        }
    }

    handleClick(event) {
        // Raycaster
        const mouse = new THREE.Vector2();
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        updateFaceExpression(vrm, 'happy', 1.0);

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

    playSound(sound) {
        if (sound.isPlaying) sound.stop();
        sound.play();
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        const delta = this.clock.getDelta();

        if (this.mixer) this.mixer.update(delta);
        this.controls.update();

        this.renderer.render(this.scene, this.camera);
    }

    updateFaceExpression(vrm, expressionName, weight) {
        const expressionManager = vrm.expressionManager;
        if (expressionManager && expressionManager.setValue) {
            expressionManager.setValue(expressionName, weight);
            expressionManager.update();
        }
    }
}

// Инициализация приложения
const viewer = new ModelViewer();
viewer.animate();