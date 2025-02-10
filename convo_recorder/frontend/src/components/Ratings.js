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

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5001';

function Ratings({ nextPage, stimOrder }) {
    const [trial, setTrial] = useState(0);
    const [currStim, setCurrStim] = useState("");

    const nextTrial = () => {
        setTrial(trial + 1);
    };

    useEffect(() => {
        // Start recording when component mounts
        fetch(`${BACKEND_URL}/start_recording`, {
            method: "POST",
        }).catch(error => console.error("Error starting recording:", error));

        // Stop recording when component unmounts
        return () => {
            fetch(`${BACKEND_URL}/stop_recording`, {
                method: "POST",
            }).catch(error => console.error("Error stopping recording:", error));
        };
    }, []);

    const updateImage = async (imageId) => {
        try {
            const response = await fetch(`${BACKEND_URL}/update_image`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ image_id: imageId }),
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        } catch (error) {
            console.error("Error updating image:", error);
        }
    };

    useEffect(() => {
        if (trial === stimOrder.length) {
            nextPage();
        } else {
            const newStim = stimOrder[trial];
            setCurrStim(newStim);
            updateImage(newStim);
        }
    }, [trial, nextPage, stimOrder]);

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
