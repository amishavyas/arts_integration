import React, { useState, useEffect } from "react";
import { Container, Typography, LinearProgress } from "@mui/material";
import { StyledButton } from "../StyledElements";
import styled from "styled-components";

const Img = styled.img`
    height: 500px;
    width: 500px;
    display: block;
    margin: auto;
    padding-top: 30px;
`;

function Ratings({ nextPage, stimOrder }) {
    const [trial, setTrial] = useState(0);
    const [currStim, setCurrStim] = useState("");

    const nextTrial = () => {
        setTrial(trial + 1);
    };

    const updateFile = async () => {
        try {
            const response = await fetch("/write_img_data", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ img: currStim }),
            });
            const result = await response.text();
            console.log(result);
        } catch (error) {
            console.error(error);
        }
    };

    useEffect(() => {
        /* Runs only when trial number updates */
        if (trial === stimOrder.length) {
            nextPage();
        } else {
            setCurrStim(stimOrder[trial]);
        }
    }, [trial]);

    useEffect(() => {
        updateFile();
    }, [currStim])

    return (
        <div>
            <LinearProgress
                variant="determinate"
                value={(trial / stimOrder.length) * 100}
                sx={{
                    height: 15,
                    backgroundColor: `#c7d1bc`,
                    "& .MuiLinearProgress-bar": {
                        backgroundColor: `#165806`,
                    },
                }}
            />
            <Container component="main" maxWidth="md" align="center">
                <Typography variant="h4" paddingTop="5%">
                    Discuss the art with your partner. Press button to continue.
                </Typography>

                <Img src={`/images/${currStim}`} />

                <StyledButton handleClick={nextTrial} text="Next Trial" />
            </Container>
        </div>
    );
}

export default Ratings;
