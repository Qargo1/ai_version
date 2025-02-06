import * as THREE from 'three/webgpu';

// Менеджер для эмоций
export default class EmotionManager {
    constructor(vrm) {
    this.vrm = vrm;
    }
    
    updateEmotion({ emotionName, value }) {
    if (this.vrm.expressionManager) {
        const clampedValue = Math.max(0, Math.min(1, value));
        console.log(`Обновление эмоции "${emotionName}" значением ${clampedValue}`);
        this.vrm.expressionManager.setValue(emotionName, clampedValue);
        this.vrm.expressionManager.update();
    } else {
        console.warn("VRM expression manager недоступен.");
    }
    }
    
    logAllExpressions() {
    if (this.vrm && this.vrm.expressionManager) {
        console.log("Expression manager structure:", this.vrm.expressionManager);
        const nameParameters = ["blinkExpressionNames", "lookAtExpressionNames", "mouthExpressionNames"];
        nameParameters.forEach(param => {
        try {
            const expressions = this.vrm.expressionManager[param];
            if (expressions && Array.isArray(expressions)) {
            console.log(`Available ${param}:`, expressions);
            expressions.forEach(expression => {
                try {
                const value = this.vrm.expressionManager.getValue(expression);
                console.log(`${expression}: ${value}`);
                } catch (error) {
                console.warn(`Failed to get value for expression "${expression}":`, error);
                }
            });
            } else {
            console.warn(`Parameter "${param}" is not an array or undefined.`);
            }
        } catch (error) {
            console.warn(`Failed to access parameter "${param}":`, error);
        }
        });
    } else {
        console.warn("VRM expression manager недоступен.");
    }
    }
    
    fixEyeTextures() {
    if (!this.vrm) return;
    const leftEyeMat = this.vrm.scene.getObjectByName("EyeLeft")?.material;
    if (leftEyeMat) {
        leftEyeMat.map = leftEyeMat.userData.originalMap || leftEyeMat.map;
        leftEyeMat.needsUpdate = true;
    }
    const rightEyeMat = this.vrm.scene.getObjectByName("EyeRight")?.material;
    if (rightEyeMat) {
        rightEyeMat.map = rightEyeMat.userData.originalMap || rightEyeMat.map;
        rightEyeMat.needsUpdate = true;
    }
    }
}