import React from "react";
import { Container, Typography } from "@mui/material";
import { StyledBox, Title } from "../StyledElements";

function Debrief() {
    return (
        <Container component="main" maxWidth="md" align="center">
            <Title text="Thank You!" />

            <Typography component="h2" variant="h6">
                <br />
                <strong> You have completed the study.</strong>
                <br />
                <p>Please ring the doorbell to alert the researchers, and they'll be
                    with you momentarily!</p> 
                <br />
            </Typography>
        </Container>
    );
}

export default Debrief;
