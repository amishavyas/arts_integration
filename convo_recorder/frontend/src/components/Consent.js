import React from "react";
import { Container, Typography } from "@mui/material";
import { Logos, StyledButton, Title } from "../StyledElements";

const Consent = ({ nextPage }) => {
    return (
        <Container component="main" maxWidth="md" align="center">
            <Logos />
            <Title text="The Art of Conversation" />

            <Typography component="h2" variant="h6">
                <br />
                <strong> Dartmouth College </strong>
                <br />
                <i>Principal Investigator</i>: Mark Allen Thornton
                <br />
                <br />
                You are being asked to take part in a{" "}
                <strong>research study</strong>. Taking part in research is{" "}
                <strong>voluntary</strong>.
            </Typography>

            <Typography>
                <br/><br/>
                <strong>CONSENT</strong>
                <br/>
                By pressing the button below, you agree that:
                <li>You agree to take part in this research.</li>
                <li>You feel like you understand what you are agreeing to.</li>
                <li>You know you are free to withdrawal at any time.</li>
            </Typography>

            <StyledButton handleClick={nextPage} text="BEGIN STUDY" />
        </Container>
    );
};

export default Consent;
