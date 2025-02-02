import * as THREE from 'three/webgpu';

// Менеджер для анимаций костей
export default class BoneAnimationManager {
    constructor(mixer, vrm) {
        this.mixer = mixer;
        this.vrm = vrm;
        this.boneLogCounters = {}
    }

    updateBoneStructure(boneData) {
        if (!this.vrm) {
            console.warn("VRM model is not loaded.");
            return;
        }
        const { boneName, position, rotation } = boneData;
        const bone = this.vrm.humanoid.getRawBoneNode(boneName);
    
        if (!bone) {
            // Логируем ошибку только первые несколько раз
            if (!this.boneErrorCounts) {
                this.boneErrorCounts = {};
            }
            if (!this.boneErrorCounts[boneName]) {
                this.boneErrorCounts[boneName] = 0;
            }
            this.boneErrorCounts[boneName]++;
            if (this.boneErrorCounts[boneName] <= 4) {
                console.warn(`Кость "${boneName}" не найдена. Ошибка №${this.boneErrorCounts[boneName]}`);
            }
            return;
        }
    
        // Обновляем позицию и поворот кости
        bone.position.set(position.x, position.y, position.z);
        bone.quaternion.set(rotation.x, rotation.y, rotation.z, rotation.w);
    
        // Отмечаем кость как обновленную
        bone.updateMatrixWorld(true);

        // Увеличиваем счетчик для логирования
        if (!this.boneLogCounters[boneName]) {
            this.boneLogCounters[boneName] = 0; // Инициализируем счетчик для кости
        }
        this.boneLogCounters[boneName]++;
    
        // Выводим лог только для каждой 10-й записи
        if (this.boneLogCounters[boneName] % 20 === 0) {
            console.log(`Обновлена кость "${boneName}":`, {
                position: {
                    x: position.x.toFixed(4),
                    y: position.y.toFixed(4),
                    z: position.z.toFixed(4)
                },
                rotation: {
                    x: rotation.x.toFixed(4),
                    y: rotation.y.toFixed(4),
                    z: rotation.z.toFixed(4),
                    w: rotation.w.toFixed(4)
                }
            });
        }
    }
}