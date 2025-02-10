import React from "react";
import { Container, Typography } from "@mui/material";
import { StyledBox, Title } from "../StyledElements";

function Debrief() {
    return (
        <Container component="main" maxWidth="md" align="center">
            <Title text="DEBRIEF TO PARTICIPANTS" />

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
                <strong> COMPLETION CODE </strong>
                <StyledBox> XXX-XXX-XXX </StyledBox>
        
            </Typography>

            <Typography component="h2" variant="h6" align="left">
                <br />
                <strong> What was the purpose of this study?</strong>
                <br />
                This study aimed to understand how people learn about, make
                sense of, and predict other people’s personality traits, mental
                states, actions, and social situations. You may not have seen
                all of these types of information, but across participants we
                presented each different type through a variety of different
                stimuli, including images, videos, audio, and text. We will
                analyze the responses that you provided to shed light on how the
                mind organizes social knowledge.
                <br />
                <br />
                <strong>What were our hypotheses?</strong>
                <br />
                We hypothesize that the mind organizes social knowledge in a way
                that is both efficient, and useful for prediction. In this
                context, efficiency means reducing the complexity of the social
                world down to a simple yet accurate summary. Social prediction
                includes predicting how other people will think, feel, and
                behavior in the future, based on what you know about them in the
                past and present. We believe that these two goals of efficiency
                and prediction shape the way that people understand their social
                world.
                <br />
                <br />
                <strong> How was this study conducted?</strong>
                <br />
                During the study, you viewed and judged a variety of stimuli and
                rated their social properties. We will analyze these data to
                determine the underlying principles that organize and explain
                your judgements. This will reveal dimensions and process that
                shape social thought.
                <br />
                <br />
                <strong>Did we tell you everything?</strong>
                <br /> There was no deception used in this study. Our initial
                description of the study brief to avoid biasing your natural
                responses. We are thus here providing you with a more detailed
                explanation of the study.
                <br />
                <br />
                <strong>Why is this study important? </strong>
                <br /> This study aims to understand the social knowledge people
                use to navigate everyday social life. The way that people's
                minds organize that social knowledge may help explain a wide
                range of important behaviors. For example, some social biases
                are thought to arise in part because the mind oversimplifies
                aspects of the social world. If we can understand how these
                processes play out, we may learn to more accurately predict and
                change them.
                <br />
                <br />
                <strong>What if I want to learn more?</strong>
                <br /> You can read more about our research - including feeling
                available copies of our published scientific articles - on our
                lab website: http://scraplab.org/
                <br /> <br /> This study was conducted by researchers at the
                Social Computation Representation And Prediction Lab in the
                Psychology and Brain Sciences Department at Dartmouth College,
                Hanover, NH, USA. If you have any questions, please contact
                principal investigator Mark Thornton at
                mark.a.thornton@dartmouth.edu.
                <br /> <br />
                This research has been reviewed and approved by the Committee
                for the Protection of Human Subjects at Dartmouth College. They
                can be reached by phone at (603) 646-6482, or by email at
                cphs@dartmouth.edu, if your questions, concerns, or complaints
                are not being answered by the research team, if you cannot reach
                the research team, if you want to talk to someone besides the
                research team, or if you have questions about your rights as a
                research participant.
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
            </Typography>
        </Container>
    );
}

export default Debrief;
