import * as tf from '@tensorflow/tfjs';
const detection = document.getElementById('detection');
const detection2 = document.getElementById('detection2');
const subirImagen = document.querySelector('#subirImagen');
const marcarImagen = document.querySelector('#marcarImagen');
const imagenRecortada = document.querySelector('#imagenRecortada');
let MODELO;

const cargarModelo = () => {
  tf.ready().then(() => {
    const modelPath = './model/tfjs_quant_uint8/model.json';
    tf.tidy(() => {
      tf.loadLayersModel(modelPath).then((model) => {
        MODELO = model;
      });
    });
  });
};

const remarcarImagen = (imagen) => {
  const myTensor = tf.browser.fromPixels(imagen);
  // Model expects 256x256 0-1 value 3D tensor
  const readyfied = tf.image
    .resizeNearestNeighbor(myTensor, [256, 256], true)
    .div(255)
    .reshape([1, 256, 256, 3]);

  const result = MODELO.predict(readyfied);
  // Model returns top left and bottom right
  result.print();

  // Draw box on canvas
  const imgWidth = imagen.width;
  const imgHeight = imagen.height;
  detection.width = imgWidth;
  detection.height = imgHeight;
  const box = result.dataSync();
  const startX = box[0] * imgWidth;
  const startY = box[1] * imgHeight;
  const width = (box[2] - box[0]) * imgWidth;
  const height = (box[3] - box[1]) * imgHeight;
  const ctx = detection.getContext('2d');
  ctx.strokeStyle = '#0F0';
  ctx.lineWidth = 4;
  ctx.strokeRect(startX, startY, width, height);

  // recortar cara de imagen
  const tHeight = myTensor.shape[0];
  const tWidth = myTensor.shape[1];
  const tStartX = box[0] * tWidth;
  const tStartY = box[1] * tHeight;
  const cropLength = parseInt((box[2] - box[0]) * tWidth, 0);
  const cropHeight = parseInt((box[3] - box[1]) * tHeight, 0);
  const startPos = [tStartY, tStartX, 0];
  const cropSize = [cropHeight, cropLength, 3];
  const cropped = tf.slice(myTensor, startPos, cropSize);
  // Prepare for next model input
  const readyFace = tf.image
    .resizeBilinear(cropped, [96, 96], true)
    .reshape([1, 96, 96, 3]);

  console.log(readyFace);

  const startX2 = 96 * imgWidth;
  const startY2 = 96 * imgHeight;
  const width2 = (box[2] - 96) * imgWidth;
  const height2 = (box[3] - 96) * imgHeight;
  const ctx2 = detection.getContext('2d');
  ctx2.strokeStyle = '#1A6D2D';
  ctx2.lineWidth = 4;
  ctx2.strokeRect(startX2, startY2, width2, height2);
};

document.addEventListener('DOMContentLoaded', () => {
  cargarModelo();

  // marcar el canvas
  document.getElementById('marcarImagen').addEventListener('click', () => {
    const petImage = document.getElementById('imagen');
    // console.log(petImage);
    remarcarImagen(petImage);
  });

  subirImagen.addEventListener('change', async function (e) {
    // obtengo el archivo
    const imagenSubida = e.target.files[0];

    // verifico que este creado el elemento img sino lo creo
    const img = document.getElementById('imagen')
      ? document.getElementById('imagen')
      : document.createElement('img');

    // le aplico sus atributos y agrego la url de la imagen
    img.setAttribute('crossorigin', 'anonymous');
    img.setAttribute('id', 'imagen');
    img.width = 300;
    img.height = 300;
    img.src = URL.createObjectURL(imagenSubida);

    // agrego la imagen al contenedor
    previsualizarImagen.appendChild(img);
    // habilito el boton
    marcarImagen.disabled = false;
    // limpio el contenedor de los datos de tensores
    // memoriaUsada.innerHTML = '';
  });
});
