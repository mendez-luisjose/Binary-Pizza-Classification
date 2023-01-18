const weight = 500;
const height = 380;

let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let otrocanvas = document.getElementById("otrocanvas");
let ctx = canvas.getContext("2d");
let currentStream = null;
let facingMode = "user";
let button = document.getElementById("guess-button");

const pizzaResultField = document.querySelector(".pizza-result");
const notPizzaResultField = document.querySelector(".notpizza-result");

let modelo = null;

class L2 {

  static className = 'L2';

  constructor(config) {
     return tf.regularizers.l1l2(config)
  }
}

tf.serialization.registerClass(L2);

//Loading the Model
(async() => {
  console.log("Loading Model...");
  modelo = await tf.loadLayersModel("./model_js/model.json");
  console.log("Model Loaded");
})();

 
window.onload = function() {
  showCamera();
}

function showCamera() {
  let options = {
    audio: false,
    video: {
      width: weight, height: height
    }
  }

  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia(options)
        .then(function(stream) {
          currentStream = stream;
          video.srcObject = currentStream;
          processCamera();
        })
        .catch(function(err) {
          alert("No se pudo utilizar la camara :(");
          console.log(err);
          alert(err);
        })
  } else {
    alert("There was an error, try again");
  }
}

function processCamera() {
  ctx.drawImage(video, 0, 0, weight, height, 0, 0, weight, height);
  setTimeout(processCamera, 20);
}

function predict() {
  notPizzaResultField.style.display = "none";
  pizzaResultField.style.display = "none";

  if (modelo != null) {
    resample_single(canvas, 256, 256, otrocanvas);

    let ctx2 = otrocanvas.getContext("2d");
    let imgData = ctx2.getImageData(0,0, 256, 256);

    let arr = [];
    let arr256 = [];

    for (let p=0; p < imgData.data.length; p+= 4) {
      let red = imgData.data[p] / 255;
      let green = imgData.data[p+1] / 255;
      let blue = imgData.data[p+2] / 255;

      arr256.push([red, green, blue]);
      if (arr256.length == 256) {
        arr.push(arr256);
        arr256 = [];
      }
    }

    arr = [arr];

    let tensor = tf.tensor4d(arr);
    let resultado = modelo.predict(tensor).dataSync();

    console.log(resultado[0])

    if (resultado >= .75) {
      pizzaResultField.style.display = "flex";
    } else {
      notPizzaResultField.style.display = "flex";
    }
  }
  

  setTimeout(predict, 2500);
}

function resample_single(canvas, width, height, resize_canvas) {
  var width_source = canvas.width;
  var height_source = canvas.height;
  width = Math.round(width);
  height = Math.round(height);

  var ratio_w = width_source / width;
  var ratio_h = height_source / height;
  var ratio_w_half = Math.ceil(ratio_w / 2);
  var ratio_h_half = Math.ceil(ratio_h / 2);

  var ctx = canvas.getContext("2d");
  var ctx2 = resize_canvas.getContext("2d");
  var img = ctx.getImageData(0, 0, width_source, height_source);
  var img2 = ctx2.createImageData(width, height);
  var data = img.data;
  var data2 = img2.data;

  for (var j = 0; j < height; j++) {
      for (var i = 0; i < width; i++) {
          var x2 = (i + j * width) * 4;
          var weight = 0;
          var weights = 0;
          var weights_alpha = 0;
          var gx_r = 0;
          var gx_g = 0;
          var gx_b = 0;
          var gx_a = 0;
          var center_y = (j + 0.5) * ratio_h;
          var yy_start = Math.floor(j * ratio_h);
          var yy_stop = Math.ceil((j + 1) * ratio_h);
          for (var yy = yy_start; yy < yy_stop; yy++) {
              var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
              var center_x = (i + 0.5) * ratio_w;
              var w0 = dy * dy; //pre-calc part of w
              var xx_start = Math.floor(i * ratio_w);
              var xx_stop = Math.ceil((i + 1) * ratio_w);
              for (var xx = xx_start; xx < xx_stop; xx++) {
                  var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
                  var w = Math.sqrt(w0 + dx * dx);
                  if (w >= 1) {
                      //pixel too far
                      continue;
                  }
                  //hermite filter
                  weight = 2 * w * w * w - 3 * w * w + 1;
                  var pos_x = 4 * (xx + yy * width_source);
                  //alpha
                  gx_a += weight * data[pos_x + 3];
                  weights_alpha += weight;
                  //colors
                  if (data[pos_x + 3] < 255)
                      weight = weight * data[pos_x + 3] / 250;
                  gx_r += weight * data[pos_x];
                  gx_g += weight * data[pos_x + 1];
                  gx_b += weight * data[pos_x + 2];
                  weights += weight;
              }
          }
          data2[x2] = gx_r / weights;
          data2[x2 + 1] = gx_g / weights;
          data2[x2 + 2] = gx_b / weights;
          data2[x2 + 3] = gx_a / weights_alpha;
      }
  }
  ctx2.putImageData(img2, 0, 0);
}


button.addEventListener("click", () => { 
  predict();
});

