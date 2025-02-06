import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, MToonMaterialLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { MToonNodeMaterial } from '@pixiv/three-vrm/nodes';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js'

import AudioManager from "audio_manager";

const MODEL_PATH = '../models/visual/vrm_models/Diamond.vrm'
const PATH_TO_ANIMATION = "../static/animations/stand_relaxed.json"

class ModelViewer {
    constructor() {
        this.clock = new THREE.Clock();
        this.initScene();
        this.initControls();
        this.initAudio();
        this.setupEventListeners();
        this.setupDragHandlers();
        // Изначально эти свойства пока null:
        this.mixer = null;
        this.vrm = null;

        this.isAnimating = true; // Флаг для управления анимацией
        this.isAnimationData = true; // Флаг для отслеживания, загрузилась ли анимация

        // Группа для модели:
        this.modelGroup = new THREE.Group();
        this.scene.add(this.modelGroup);

        // Загрузчики
        this.gltfLoader = new GLTFLoader();
        this.fbxLoader = new FBXLoader();

        // Загружаем модель
        this.loadModel();

        // Инициализация списка эмоций
        this.EMOTIONS = [
            "aa", "ih", "ou", "ee", "oh", "blink", "blinkLeft", "blinkRight",
            "neutral", "sad", "angry", "happy", "surprised", "lookLeft", "lookRight", 
            "lookUp", "lookDown"
        ];

        // Начать случайную смену эмоций каждые 5 секунд
        this.changeEmotionRandomly();
    }

    // Функция для случайной смены эмоции
    getRandomEmotion() {
        const randomIndex = Math.floor(Math.random() * this.EMOTIONS.length);
        return this.EMOTIONS[randomIndex];
    }

    // Функция для обновления эмоции
    changeEmotion() {
        // Получаем доступ к Expression Manager
        const expressionManager = this.vrm.expressionManager;

        // Пример выбора случайного выражения из доступных
        const expressions = ["happy", "sad", "angry", "surprised", "blink"];
        const randomExpression = expressions[Math.floor(Math.random() * expressions.length)];

        // Устанавливаем выражение
        expressionManager.setValue(randomExpression, 1);  // Установить выражение в 100% интенсивности
        console.log("randomExpression is", randomExpression); 
    }

    // Функция для вызова смены эмоции каждые 5 секунд
    changeEmotionRandomly() {
        setInterval(() => {
            this.changeEmotion();
        }, 5000); // 5000 миллисекунд = 5 секунд
    }

    initAudio() {
        this.audioManager = new AudioManager(this.camera);
        // Пример загрузки звука клика:
        this.audioManager.loadSound('click', 'http://127.0.0.1:5500/tools/sounds/clicking.mp3', {
            volume: 0.5,
            loop: false
        });
    }

    animate = () => {
        if (!this.isAnimating) {
            return; // Прекращаем анимацию, если флаг установлен в false
        }
    
        requestAnimationFrame(this.animate);
    
        if (this.mixer && typeof this.mixer.update === 'function') {
            this.mixer.update(this.clock.getDelta());
        } else {
            console.error("No mixer found. Ensure you've loaded a VRM model and animations.");
            this.stopAnimation(); // Останавливаем анимацию
            return;
        }
    
        this.renderer.renderAsync(this.scene, this.camera);
    };

    startAnimation() {
        if (this.mixer && typeof this.mixer.update === 'function') {
            this.isAnimating = true; // Включаем флаг
            console.log("Animation started.");
            this.animate(); // Запускаем анимацию
        } else {
            console.warn("Cannot start animation: Mixer is not initialized.");
        }
    }

    stopAnimation() {
        this.isAnimating = false; // Отключаем флаг
        console.log("Animation stopped.");
    }

    initScene() {
        const canvas = document.createElement('canvas');
        document.body.appendChild(canvas);

        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 1, -3);
        this.camera.lookAt(0, 0, 0);

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xeeeeee);

        try {
            this.renderer = new THREE.WebGPURenderer({
                canvas: document.querySelector('canvas'),
                antialias: true,
                alpha: true,
            });
            this.renderer.init().then(() => {
                this.renderer.setSize(window.innerWidth, window.innerHeight);
                this.renderer.setPixelRatio(window.devicePixelRatio);
                this.renderer.shadowMap.enabled = true;
            }).catch(error => console.error("Renderer init failed:", error));
        } catch (error) {
            console.warn("WebGPU is not supported. Falling back to WebGL.");
            this.renderer = new THREE.WebGLRenderer({
                canvas: document.querySelector('canvas'),
                antialias: true,
                alpha: true,
            });
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.setPixelRatio(window.devicePixelRatio);
            this.renderer.shadowMap.enabled = true;
        }

        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        this.directionalLight.position.set(5, 5, 5);
        this.directionalLight.castShadow = true;
        this.scene.add(this.ambientLight, this.directionalLight);
    }

    initControls() {
        if (!this.camera || !this.renderer) {
            console.error("Camera or renderer is not initialized");
            return;
        }
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 1;
        this.controls.minDistance = 0.01;
        this.controls.maxDistance = 7;
    }

    async loadAnimationData(filePath) {
        try {
            const response = await fetch(filePath);
            return await response.json();
        } catch (err) {
            console.log(`Failed to load animation data: ${err}`);
            return null;
        }
        
    }
    
    createAnimationClips(animationData, vrm) {
        const clips = [];
        
        if (!vrm || !vrm.humanoid) {
            console.error("VRM or humanoid missing. Cannot proceed with animation.");
            return clips; // Возвращаем пустой массив, если данных нет
        }
    
        for (const boneData of animationData.bones) {
            const boneName = boneData.name;
            const bone = vrm.humanoid.getRawBoneNode(boneName);
    
            if (!bone) {
                console.warn(`Bone "${boneName}" not found.`);
                continue;
            }
    
            const times = boneData.keyframes.map(frame => frame.time);
            const positions = boneData.keyframes.flatMap(frame => frame.position);
            const rotations = boneData.keyframes.flatMap(frame => frame.rotation);
    
            const positionTrack = new THREE.VectorKeyframeTrack(
                `${bone.name}.position`,
                times,
                positions
            );
            const rotationTrack = new THREE.QuaternionKeyframeTrack(
                `${bone.name}.quaternion`,
                times,
                rotations
            );
    
            const clip = new THREE.AnimationClip(boneName, -1, [positionTrack, rotationTrack]);
            clips.push(clip);
        }
    
        return clips;
    }
    
    async setupAnimation(vrm, animationData) {
        if (!vrm || !vrm.humanoid) {
            console.error("Humanoid or VRM model is not properly loaded.");
            return;
        }
    
        const clips = this.createAnimationClips(animationData, vrm);
    
        if (!this.mixer) {
            // Создаем новый mixer, если его нет
            this.mixer = new THREE.AnimationMixer(vrm.scene);
            console.log("Mixer created");
        } else {
            // Если mixer уже существует, очищаем его перед загрузкой новой анимации
            this.mixer.stopAllAction();
            console.log("Mixer emptied out");
        }
    
        // Загружаем все анимации в mixer
        for (const clip of clips) {
            const action = this.mixer.clipAction(clip);

            // Измените скорость анимации
            this.mixer.clipAction(clip).setEffectiveTimeScale(20.0);  // Ускоряет анимацию в 2 раза
            

            action.setLoop(THREE.LoopRepeat);
            action.play();
        }
    
        // Теперь можно запустить анимацию
        this.startAnimation();
    
        return this.mixer;
    }

    // Переключение анимации - не добавлено
    switchAnimation(newAnimationClip) {
        if (this.mixer) {
            this.mixer.stopAllAction(); // Останавливаем текущие анимации
    
            const action = this.mixer.clipAction(newAnimationClip);
            action.setLoop(THREE.LoopRepeat);
            action.play();
        }
    }

    async loadModel() {
        if (!MODEL_PATH) {
            console.error("Model path not defined.");
            return;
        }
        try {
            console.log(`Loading model from: ${MODEL_PATH}`);
            const response = await fetch(MODEL_PATH);
            if (!response.ok) throw new Error(`Failed to load model: ${MODEL_PATH}`);
            const arrayBuffer = await response.arrayBuffer();

            const loader = new GLTFLoader();
            loader.crossOrigin = 'anonymous';
            loader.register(parser => {
                const mtoonMaterialPlugin = new MToonMaterialLoaderPlugin(parser, {
                    materialType: MToonNodeMaterial,
                });
                return new VRMLoaderPlugin(parser, {
                    mtoonMaterialPlugin,
                });
            });

            loader.parse(arrayBuffer, '', async (gltf) => {
                if (gltf.userData && gltf.userData.vrm) {
                    this.vrm = gltf.userData.vrm;
                    this.modelGroup.add(this.vrm.scene);

                    // Устанавливаем начальное положение модели
                    this.vrm.scene.position.set(0.05, -1, 0);  // Пример: перемещение модели на 1 единицу вверх по оси Y

                    // Если анимация не загружена, пропускаем настройку анимации
                    if (this.isAnimationData) {
                        const animationData = await this.loadAnimationData(PATH_TO_ANIMATION);
                        if (!animationData) {
                            console.error("Failed to load animation data.");
                            return;
                        }
                        this.mixer = await this.setupAnimation(this.vrm, animationData);
                        console.log("mixer loaded successfully", this.mixer);
                    } else {
                        console.log("Animation data not found, skipping animation setup.");
                    }

                } else {
                    console.error("VRM not found in the model.");
                }
            }, error => console.error("Error parsing model:", error));
        } catch (error) {
            console.error("Preloading failed:", error);
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
            this.controls.enabled = false;
            this.isDragging = true;
            this.dragStartMouse.set(event.clientX, event.clientY);
            this.dragStartModelPosition.copy(clickedObject.position);
            this.draggedObject = clickedObject;
        } else {
            console.warn("No object was clicked.");
        }
    }

    onMouseUp() {
        this.isDragging = false;
        this.draggedObject = null;
        this.controls.enabled = true;
    }

    onMouseMove(event) {
        if (!this.isDragging || !this.draggedObject) return;

        const scaleFactor = 0.001;
        const deltaX = (event.clientX - this.dragStartMouse.x) * scaleFactor;
        const deltaY = (event.clientY - this.dragStartMouse.y) * scaleFactor;

        this.draggedObject.position.x = this.dragStartModelPosition.x + deltaX;
        this.draggedObject.position.y = this.dragStartModelPosition.y - deltaY;
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize());
        window.addEventListener('click', (e) => {
            this.handleClick(e);
            if (THREE.AudioContext) {
                THREE.AudioContext.getContext().resume();
            }
        }, { once: true });

        document.addEventListener('keydown', (e) => {
            if (e.key === ' ') this.toggleAnimation();
        });
    }

    toggleAnimation() {
        if (this.mixer) {
            if (this.isAnimationPlaying) {
                this.mixer.stopAllAction();
            } else {
                this.mixer.update(this.clock.getDelta());  // Включаем анимацию
            }
            this.isAnimationPlaying = !this.isAnimationPlaying;
        }
    }    

    setObjectMaterial(object, materialType) {
        const material = new MToonNodeMaterial();
        if (materialType === 'shinny') {
            material.color.set(0xffffff);
            material.shininess = 50;
        } else if (materialType === 'matte') {
            material.color.set(0xaaaaaa);
            material.shininess = 5;
        } else {
            material.color.set(0x888888);
        }
        object.material = material;
    }    

    handleClick(event) {
        const mouse = new THREE.Vector2();
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        const intersects = raycaster.intersectObjects(this.modelGroup.children, true);
        if (intersects.length > 0) {
            const object = intersects[0].object;
            this.highlightObject(object);
        }
    }

    createUI() {
        const button = document.createElement('button');
        button.innerText = 'Start Animation';
        button.addEventListener('click', () => this.toggleAnimation());
        document.body.appendChild(button);
    }    

    loadObjectIfNeeded(object) {
        const distance = this.camera.position.distanceTo(object.position);
        if (distance < 5 && !object.isLoaded) {
            this.loadObject(object);
            object.isLoaded = true;
        }
    }
    

    toggleCameraMode() {
        this.controls.enableZoom = !this.controls.enableZoom;
        this.controls.enablePan = !this.controls.enablePan;
    }

    resetCameraPosition() {
        this.camera.position.set(0, 1, -3);
        this.camera.lookAt(0, 0, 0);
        this.controls.update();
    }
    

    highlightObject(object) {
        if (object.material) {
            object.material.color.setHex(Math.random() * 0xffffff);
        } else {
            console.warn("Clicked object does not have a material.");
        }
    }    

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

// Инициализация приложения
const viewer = new ModelViewer();