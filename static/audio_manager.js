import * as THREE from 'three/webgpu';

// Менеджер для звука
export default class AudioManager {
    constructor(camera) {
        this.audioListener = new THREE.AudioListener();
        camera.add(this.audioListener);
        this.sounds = {};
    }

    loadSound(name, url, options = {}) {
        const sound = new THREE.Audio(this.audioListener);
        const loader = new THREE.AudioLoader();
        loader.load(
            url,
            (buffer) => {
                sound.setBuffer(buffer);
                sound.setVolume(options.volume !== undefined ? options.volume : 1.0);
                sound.setLoop(options.loop !== undefined ? options.loop : false);
                if (options.autoplay) sound.play();
            },
            undefined,
            (error) => {
                console.error(`Ошибка загрузки звука "${name}" с URL "${url}":`, error);
            }
        );
        this.sounds[name] = sound;
        return sound;
    }

    playSound(name) {
        const sound = this.sounds[name];
        if (!sound) {
            console.warn(`Звук "${name}" не найден.`);
            return;
        }
        if (sound.isPlaying) sound.stop();
        sound.play();
    }

    stopSound(name) {
        const sound = this.sounds[name];
        if (sound && sound.isPlaying) sound.stop();
    }

    getSound(name) {
        return this.sounds[name];
    }
}