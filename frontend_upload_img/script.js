let button = document.getElementById("guess-button");
const canvasAn = document.getElementById("canvas-an");
let context = canvasAn.getContext("2d");
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
  console.log("Model Loaded!");
})();

function predict() {
  notPizzaResultField.style.display = "none";
  pizzaResultField.style.display = "none";

  if (modelo != null) {
    let ctx2 = canvasAn.getContext("2d");
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
    let result = modelo.predict(tensor).dataSync();

    if (result >= .75) {
      pizzaResultField.style.display = "flex";
    } else {
      notPizzaResultField.style.display = "flex";
    }      
  }
}

const loadImage = () => {
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("preview").setAttribute("src", e.target.result)
  };
  reader.readAsDataURL(input.files[0]);
};

const setImage = () => {
  const img = document.getElementById("preview");
  context.drawImage(img, 0, 0, 300, 300);
};

button.addEventListener("click", () => {
  setImage();
  predict();
});

