import React from "react";
import {
    Container,
    Checkbox,
    Typography,
    RadioGroup,
    Radio,
    FormGroup,
    FormControlLabel,
    TextField,
} from "@mui/material";
import { Logos, StyledButton, Title } from "../StyledElements";

const raceLabels = [
    "American Indian or Alaska Native",
    "Black or African American",
    "South Asian",
    "South East Asian",
    "East Asian",
    "Middle Eastern / North African",
    "Native Hawaiian or Other Pacific Islander",
    "White",
    "Other",
    "Prefer not to answer",
];
const genderLabels = [
    "Man",
    "Woman",
    "Non-binary",
    "Other",
    "Prefer not to answer",
];
const educationLabels = [
    "Less than high school credential",
    "High school credential",
    "Some post-high-school, no bachelor's degree",
    "Bachelor's degree",
    "Graduate degree",
    "Other",
    "Prefer not to answer",
];
const sexLabels = ["Female", "Male", "Other", "Prefer not to answer"];
const ethnicityLabels = ["Hispanic or Latino", "Not Hispanic or Latino"];

function DemoSurvey({ nextPage, demoData, setDemoData }) {
    const submitDemoSurvey = () => {
        if (validateResponses()) {
            sendData(); 
            nextPage();
        }
    };

    const sendData = async () => {
        try {
            const response = await fetch("/write_to_db", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify([{ 'pairID': "1", 'subID': "2", 'imgID': 10, 'audio_path': "some_path", 'embedding': [1,2,3,4,5] }]),
            });
            console.log(
                JSON.stringify({
                    imgID: 10,
                    audio_path: "some_path",
                    embedding: [1, 2, 3, 4, 5],
                }), typeof(10)
            );
            const result = await response.text();
            console.log(result);
        } catch (error) {
            console.error(error);
        }
    };

    const validateResponses = () => {
        for (const value in demoData) {
            if (demoData[value] === "" || demoData[value].length === 0) {
                alert("Please enter your response for " + value.toUpperCase());
                return false;
            }
            if (demoData["age"] < 18) {
                alert("Please enter your age correctly");
                return false;
            }
        }
        return true;
    };

    const handleRaceSelection = (e) => {
        if (e.target.checked) {
            if (!demoData["race"].includes(e.target.value)) {
                setDemoData({
                    ...demoData,
                    race: [...demoData["race"], e.target.value],
                });
            }
        } else if (!e.target.checked) {
            demoData["race"].splice(
                demoData["race"].indexOf(e.target.value),
                1
            );
            setDemoData({ ...demoData, race: demoData["race"] });
        }
    };

    return (
        <div>
            <Container component="main" maxWidth="md" align="center">
                <Logos />
                <Title text="DEMOGRAPHIC SURVEY" />
                <br />
                <Typography fontSize="21px">
                    You have finished all the trials! <br />
                    <br />
                    Please complete the following survey. You will be debriefed
                    and receive your completion code on the next page.
                </Typography>

                <br />

                <Typography align="left" component="h1" variant="h6">
                    <strong> Please enter your age</strong>
                    <TextField
                        label=""
                        placeholder="Age"
                        type="number"
                        InputProps={{ style: { fontSize: 19 } }}
                        variant="standard"
                        color="success"
                        fullWidth
                        onChange={(e) =>
                            setDemoData({ ...demoData, age: e.target.value })
                        }
                    />
                    <br /> <br />
                    <strong>Please indicate your gender</strong>
                    <br />
                    <RadioGroup
                        name="controlled-radio-buttons-group"
                        onChange={(e) =>
                            setDemoData({ ...demoData, gender: e.target.value })
                        }
                        inputlabelprops={{ style: { fontSize: 18 } }}
                    >
                        {genderLabels.map((gender, index) => (
                            <FormControlLabel
                                value={gender}
                                key={index}
                                control={<Radio color="success" />}
                                label={
                                    <Typography style={{ fontSize: "19px" }}>
                                        {" "}
                                        {gender}{" "}
                                    </Typography>
                                }
                            />
                        ))}
                    </RadioGroup>
                    <br />
                    <strong> Please indicate your sex at birth </strong>
                    <RadioGroup
                        name="controlled-radio-buttons-group"
                        onChange={(e) =>
                            setDemoData({ ...demoData, sex: e.target.value })
                        }
                        inputlabelprops={{ style: { fontSize: 18 } }}
                    >
                        {sexLabels.map((sex, index) => (
                            <FormControlLabel
                                value={sex}
                                key={index}
                                control={<Radio color="success" />}
                                label={
                                    <Typography style={{ fontSize: "19px" }}>
                                        {" "}
                                        {sex}{" "}
                                    </Typography>
                                }
                            />
                        ))}
                    </RadioGroup>
                    <br />
                    <strong>
                        Please indicate your race. Select all that apply.
                    </strong>
                    <FormGroup onChange={handleRaceSelection}>
                        {raceLabels.map((race, index) => (
                            <FormControlLabel
                                value={race}
                                key={index}
                                control={<Checkbox color="success" />}
                                label={
                                    <Typography style={{ fontSize: "19px" }}>
                                        {race}
                                    </Typography>
                                }
                            />
                        ))}
                    </FormGroup>
                    <br />
                    <strong>Please indicate your ethnicity</strong>
                    <RadioGroup
                        onChange={(e) =>
                            setDemoData({
                                ...demoData,
                                ethnicity: e.target.value,
                            })
                        }
                    >
                        {ethnicityLabels.map((ethnicity, index) => (
                            <FormControlLabel
                                value={ethnicity}
                                key={index}
                                control={<Radio color="success" />}
                                label={
                                    <Typography style={{ fontSize: "19px" }}>
                                        {ethnicity}
                                    </Typography>
                                }
                            />
                        ))}
                    </RadioGroup>
                    <br />
                    <strong>
                        {" "}
                        What is the highest level of education you have
                        completed?
                    </strong>
                    <RadioGroup
                        name="controlled-radio-buttons-group"
                        onChange={(e) =>
                            setDemoData({
                                ...demoData,
                                education: e.target.value,
                            })
                        }
                    >
                        {educationLabels.map((educationLevel, index) => (
                            <FormControlLabel
                                value={educationLevel}
                                key={index}
                                control={<Radio color="success" />}
                                label={
                                    <Typography style={{ fontSize: "19px" }}>
                                        {educationLevel}
                                    </Typography>
                                }
                            />
                        ))}
                    </RadioGroup>
                </Typography>

                <StyledButton handleClick={submitDemoSurvey} text="SUBMIT" />
            </Container>
        </div>
    );
}

export default DemoSurvey;
