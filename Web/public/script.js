
let arr = []


function displayImage(src) {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = function () {
        canvas.width = 28;
        canvas.height = 28;
        ctx.drawImage(img, 0, 0, 28, 28);
        const imageData = ctx.getImageData(0, 0, 28, 28);
        const data = imageData.data;
        const grayscaleArray = [];

        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const grayscale = 0.299 * r + 0.587 * g + 0.114 * b;
            grayscaleArray.push(grayscale);
        }

        for (let i = 0; i < grayscaleArray.length; i++) {
            const grayscale = Math.max(0, Math.min(255, grayscaleArray[i]));
            singleGrid[i].style.backgroundColor = `rgb(${grayscale}, ${grayscale}, ${grayscale})`;
        }
        arr = grayscaleArray;
        console.log(grayscaleArray);
    };
    img.src = src;
}

const canvas = document.querySelector(".canvas");

for (let i = 0; i < 28 * 28; i++) {
    canvas.innerHTML += `  <div class="single-grid"></div>`
}

const singleGrid = document.querySelectorAll(".single-grid")
const result = document.querySelector(".result")
const resutArr = document.querySelector(".result-arr")

// let isMouseDown = false;
// canvas.addEventListener("mousedown", () => {
//     isMouseDown = true;
// });
// document.addEventListener('mouseup', () => {
//     isMouseDown = false;
// });
// if (singleGrid.length == 28 * 28) {
//     for (let i = 0; i < singleGrid.length; i++) {
//         arr.push(0)
//         element = singleGrid[i];
//         singleGrid[i].addEventListener("mouseover", () => {

//             if (isMouseDown && !singleGrid[i].classList.contains("on")) {
//                 singleGrid[i].classList.add("on")
//                 arr[i] = 255;
//             }
//         })
//     }
// } else {
//     alert("Something Wrong!")
// }

const btn = document.querySelector(".print-btn");
const load = document.querySelector(".load-btn");

load.addEventListener("click", () => {
    const myInterval = setInterval(() => {
        displayImage("test/" + Math.floor(Math.random() * 149) + ".png")
    }, 50);

    setTimeout(() => {
        clearInterval(myInterval)
    }, 400);
});
btn.addEventListener("click", () => {
    buttonPressed();
    sendData(arr)
})

function buttonPressed() {

}
function sendData(data) {
    fetch("http://localhost:3000/pridict", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ data: data }) // Stringify object with userid
    }).then(response => response.json())
        .then(data => {
            result.innerHTML = predictDigit(data.pridiction);
            resutArr.innerHTML = mapResult(data.pridiction);
        })
        .catch(error => console.error(error));
}
function mapResult(result) {
    el = ``
    for (let i = 0; i < result.length; i++) {
        el += `<p class="result-pro"> ${i} , ${result[i].toFixed(2)}</p>`
    }
    return el;
}
function predictDigit(probabilities) {
    let maxIndex = 0;
    let maxValue = probabilities[0];

    for (let i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxValue) {
            maxIndex = i;
            maxValue = probabilities[i];
        }
    }

    return maxIndex;
}
