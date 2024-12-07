import React, { useState, useEffect } from "react";
import Consent from "./Consent";
import DemoSurvey from "./DemoSurvey";
import Ratings from "./Ratings";
import Debrief from "./Debrief";

function Experiment() {
    const [page, setPage] = useState(1);
    const stimOrder = [
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
    ];
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
                    return (
                        <DemoSurvey
                            nextPage={nextPage}
                            demoData={demoData}
                            setDemoData={setDemoData}
                        />
                    );
                case 4:
                    return <Debrief />;
                default:
            }
        }
    };

    return <div>{conditionalComponent()}</div>;
}

export default Experiment;
