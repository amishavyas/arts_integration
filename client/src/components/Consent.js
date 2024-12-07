import React from "react";
import { Container, Typography } from "@mui/material";
import { Logos, StyledButton, Title } from "../StyledElements";

const Consent = ({ nextPage }) => {
    return (
        <Container component="main" maxWidth="md" align="center">
            <Logos />
            <Title text="CONSENT TO TAKE PART IN RESEARCH" />

            <Typography component="h2" variant="h6">
                <br />
                <strong> Dartmouth College </strong>
                <br />
                <i>Study title</i>: Online behavioral studies of social
                cognition
                <br />
                <i>Principal Investigator</i>: Mark Allen Thornton
                <br />
                <br />
                You are being asked to take part in a{" "}
                <strong>research study</strong>. Taking part in research is{" "}
                <strong>voluntary</strong>.
            </Typography>

            <Typography component="h2" variant="h6" align="left">
                <br />
                <strong> Study Summary: </strong>
                <br />
                This study aims to understand how people perceive and predict
                other people and social situations. The risk of participating in
                this study is minimal, no greater than what you would experience
                in the course of everyday internet use for an equivalent period
                of time. Your participation in this study is completely
                voluntary. You many choose not to participate. You may withdraw
                at any time during the study without penalty by simply closing
                this tab or window of your web browser.
                <br />
                <br />
                <strong> What is the purpose of this study? </strong>
                <br />
                The purpose of the study is to understand how people learn about
                and make sense of other people’s traits (e.g., extraversion),
                mental states (thoughts and feelings), actions (running, eating
                talking etc.), and social situations (e.g. dinner with friends).
                We also hope to learn how these processes apply to nonhuman
                objects and entities (such as computer avatars, algorithms,
                pets, and graphs).
                <br />
                <br />
                <strong>
                    {" "}
                    Will you benefit from taking part in this study?{" "}
                </strong>
                <br />
                The only benefit you will receive from participating is the
                monetary payment described below.
                <br />
                <br />
                <strong>What does this study involve?</strong>
                <br />
                Your participation in this study will last approximately [value
                between 5 and 60 inserted here, as appropriate] minutes. You
                will begin by reading the study instructions. You will then make
                judgments about trait, mental states, actions, and/or
                situations, or nonsocial object or entities (such computer
                avatars, algorithms, pets, and graphs). The stimuli may be
                presented through a variety of different media, including text,
                images, videos, and audio. You will make your responses using
                your keyboard/mouse or touchscreen. At the end of the study, you
                will be asked to complete a brief questionnaire about yourself,
                relating information such as your demographics, personality
                traits, mental state, and attitudes.
                <br />
                <br />
                <strong>
                    What are the risks involved with being enrolled in this
                    study?{" "}
                </strong>
                <br />
                The risk of participating in this study is minimal. All stimuli
                you will be presented with are selected to avoid any sensitive
                content, such as violence. You will not be asked to provide any
                sensitive information about yourself either.
                <br />
                <br />
                <strong>
                    Will my deidentified data be shared with the public for
                    scientific purposes?{" "}
                </strong>
                <br />
                Yes, for scientific purposes, the data collected in this project
                will be shared on Open Science Framework and freely accessible
                to the public for scientific purposes. We will not record
                identifying information from you, such as names, contact
                information, or IP addresses. We may temporarily retain your
                [Mechanical Turk/Prolific, depending on which platform is being
                used] ID to ensure that we can pay you if you encounter a
                technical issue. However, this information will be deleted after
                data collection is complete, and will not be shared.
                <br />
                <br />
                <strong>Will you be paid to take part in this study?</strong>
                <br />
                Yes. You will be paid [insert $7.50/hr x duration of study].
                This payment will be made electronically via the online platform
                you are using to complete the study.
                <br />
                <br />
                <strong>
                    Whom should you call with questions about this study?
                </strong>
                <br />
                The leader of the research team for this study is Mark Thornton,
                who can be reached at mark.a.thornton@dartmouth if you have
                questions, concerns, or complaints, if you would like to talk to
                the research team, if you think the research has harmed you, or
                if you wish to withdraw from the study.
                <br />
                <br />
                This research has been reviewed and approved by the Committee
                for the Protection of Human Subjects at Dartmouth College. They
                can be reached by phone at (603) 646-6482, or by email at
                cphs@dartmouth.edu, if your questions, concerns, or complaints
                are not being answered by the research team, if you cannot reach
                the research team, if you want to talk to someone besides the
                research team, or if you have questions about your rights as a
                research participant
                <br />
                <br />
                <strong>CONSENT</strong>
                <br />
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
