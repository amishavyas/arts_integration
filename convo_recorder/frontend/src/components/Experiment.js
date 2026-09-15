import React, { useState, useEffect } from "react";
import Consent from "./Consent";
import Ratings from "./Ratings";
import Debrief from "./Debrief";

function Experiment() {
    const [page, setPage] = useState(1);
    const unshuffledStim = [
        "img_01.png",
        "img_02.png",
        "img_03.png",
        "img_04.png",
        "img_05.png",
        "img_06.png",
        "img_07.png",
        "img_08.png",
        "img_09.png",
        "img_10.png",
        "img_11.png",
        "img_12.png",
        "img_13.png",
        "img_14.png",
        "img_15.png",
        "img_16.png",
        "img_17.png",
        "img_18.png",
        "img_19.png",
        "img_20.png",
        "img_21.png",
        "img_22.png",
        "img_23.png",
        "img_24.png",
        "img_25.png"
    ];

    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
      }
 
    const stimOrder = shuffleArray(unshuffledStim);


    const [demoData, setDemoData] = useState({
        age: "",
        education: "",
        gender: "",
        sex: "",
        ethnicity: "",
        race: [],
    });

    useEffect(() => {
        // Delete file on page refresh
        fetch("/reset_img_data", {
            method: "GET",
        });
    }, []);

    const nextPage = () => {
        setPage(page + 1);

        window.scrollTo(0, 0);
    };

    const conditionalComponent = () => {
        if (page !== 0) {
            switch (page) {
                case 1:
                    return <Consent nextPage={nextPage} />;
                case 2:
                    return (
                        <Ratings nextPage={nextPage} stimOrder={stimOrder} />
                    );
                case 3:
                    return <Debrief />;
                default:
            }
        }
    };

    return <div>{conditionalComponent()}</div>;
}

export default Experiment;
