const loadFile1 = document.getElementById("loadFile1");
const inputFile = document.getElementById("inputFile");
const inputFile2 = document.getElementById("inputFile2");
const inputFile3 = document.getElementById("inputFile3");
const inputFileName = document.getElementById("inputFileName");
const loadingAnimation = document.getElementById("loadingAnimation");
const loadingText = document.getElementById("loadingText");
const classification = document.getElementById("classification");
const movieStars = document.getElementById("movieStars");
const heartEmpty = "far fa-heart";
const heartFull = "fas fa-heart";
let whichClassifier = 0;
const buttonB = document.getElementById("ClassifyB");
const buttonS = document.getElementById("ClassifyS");
const buttonK = document.getElementById("ClassifyK");

function movieClass(response) {
    response.json().then((data) => {
        console.log(data);

        loadingAnimation.classList.remove("spinner");
        loadingText.textContent = "";

        let textContent = "La Película es ";
        let good = 0;
        let bad = 0;

        switch (data.label) {
            case "Mala":
                bad = 3;
                good = 0;
                break;

            case "Regular":
                bad = 2;
                good = 1;
                break;

            case "Buena":
                bad = 1;
                good = 2;
                break;

            case "Excelente":
                bad = 0;
                good = 3;
                break;
        }

        classification.textContent = textContent + " " + data.label;

        for (let index = 0; index < good; index++) {
            let span = document.createElement("span");
            span.className = "heartColor";
            let i = document.createElement("i");
            i.className = heartFull;
            span.appendChild(i);
            movieStars.appendChild(span);
        }

        for (let index = 0; index < bad; index++) {
            let span = document.createElement("span");
            span.className = "heartColor";
            let i = document.createElement("i");
            i.className = heartEmpty;
            span.appendChild(i);
            movieStars.appendChild(span);
        }
    })
}

loadFile1.addEventListener("submit", e => {
    e.preventDefault();

    let whichEndPoint;

    switch(whichClassifier) {
        case 0:
            whichEndPoint = "Laplace";
            break;
        case 1:
            whichEndPoint = "SVM";
            break;
        case 2:
            whichEndPoint = "KNN";
            break;
    }

    console.log(whichEndPoint);


    const endpoint = `${window.origin}/${whichEndPoint}`;
    const formData = new FormData();

    formData.append("inputFile", inputFile.files[0]);

    console.log(inputFile.files[0]);

    if (inputFile.files[0] === undefined) {
        return;
    }

    movieStars.innerHTML = "";
    classification.textContent = "";
    loadingAnimation.classList.add("spinner");
    loadingText.textContent = "Cargando...";

    fetch(endpoint, {
        method: "POST",
        credentials: "include",
        body: formData,
        cache: "no-cache"
    }).then((response) => {
        movieClass(response)
    })
})

inputFile.addEventListener("change", function(){
    inputFileName.textContent = this.files[0].name;
})

buttonB.addEventListener("click", function() {
    whichClassifier = 0;
});
buttonS.addEventListener("click", function() {
    whichClassifier = 1;
});
buttonK.addEventListener("click", function() {
    whichClassifier = 2;
});
