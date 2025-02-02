// Парсинг данных о позиции кости
function parseBonePosition(rawData) {
  const [boneName, x, y, z, qx, qy, qz, qw] = rawData.args;

  // Преобразуем имя кости через карту соответствия и делаем первую букву строчной
  const mappedBoneName = boneMapping[boneName] || boneName;
  const processedBoneName = toLowerCaseFirstLetter(mappedBoneName);

  return {
    boneName: processedBoneName,
    position: { x, y, z },
    rotation: { x: qx, y: qy, z: qz, w: qw }
  };
}

// Парсинг данных о смешивании эмоций
function parseBlendValue(rawData) {
  const [emotionName, value] = rawData.args;
  return { emotionName, value };
}

// Обработчик сообщений от основного потока
self.onmessage = (event) => {
  try {
      const animationData = event.data;

      // Преобразуем имена костей и формируем команды
      const processedCommands = animationData.map(frame => {
          if (frame.address === "/VMC/Ext/Bone/Pos") {
              const parsedData = parseBonePosition(frame);
              return {
                  type: "updateModel",
                  data: parsedData
              };
          } else if (frame.address === "/VMC/Ext/Blend/Val") {
              const parsedData = parseBlendValue(frame);
              return {
                  type: "updateEmotion",
                  data: parsedData
              };
          }
          return null; // Пропускаем неизвестные типы фреймов
      }).filter(Boolean); // Удаляем null значения

      console.log("Worker processed commands:", processedCommands); // Логируем обработанные команды


      // Используем setInterval, чтобы отправлять команды по одной каждые 10 мс
      let frameIndex = 0;
      const interval = setInterval(() => {
        if (frameIndex >= processedCommands.length) {
          clearInterval(interval);
          // Можно отправить специальное сообщение, что анимация завершена
          self.postMessage({ type: "playbackCompleted" });
          return;
        }
        // Отправляем текущую команду
        self.postMessage(processedCommands);
        frameIndex++;
      }, 10);
  } catch (error) {
      console.error("Worker error:", error); // Логируем ошибки внутри Worker'а
  }
};