import { Box, Button, Slider, Typography } from "@mui/material";
import dartLogo from "./images/D-Pine_CMYK.png";
import labLogo from "./images/lab-logo.png";
import { styled } from "@mui/material/styles";
import styledComponent from "styled-components";

const Logos = () => {
    return (
        <Box
            style={{
                display: "flex",
                justifyContents: "center",
                paddingTop: "2%",
            }}
        >
            <img
                src={dartLogo}
                alt={"Dartmouth logo"}
                style={{
                    height: "150px",
                    margin: "auto",
                    marginRight: "-25%",
                }}
            ></img>
            <img
                src={labLogo}
                alt={"SCRAP Lab logo"}
                style={{
                    height: "150px",
                    margin: "auto",
                }}
            ></img>
        </Box>
    );
};

const StyledBox = styledComponent.div`
    text-align: center;
    display: flex;
    border-radius: 5px;
    background-color: #e4d09e;
    opacity: 60%;
    flex-direction: column;
    font-size: 30px;
    color: black;
    margin: auto;
    min-height: 10px;
    width: 45%;
`;

const StyledButton = ({ handleClick, text }) => {
    return (
        <Box
            style={{
                display: "flex",
                justifyContent: "center",
                padding: "5%",
            }}
        >
            <Button
                onClick={handleClick}
                variant="contained"
                style={{
                    color: "#212528",
                    fontSize: "15px",
                    backgroundColor: "#e4d09e",
                }}
            >
                {text}
            </Button>
        </Box>
    );
};

const StyledSlider = styled(Slider)({
    color: "#5a5136",
    height: 8,
    "& .MuiSlider-track": {
        border: "none",
    },
    "& .MuiSlider-thumb": {
        height: 15,
        width: 15,
        backgroundColor: "beige",
        border: "1px solid darkbeige",
        boxShadow: "lightgrey 0 1px 1px",
        "&:focus, &:hover, &.Mui-active, &.Mui-focusVisible": {
            boxShadow: "#ccc 0 2px 2px",
        },
    },
    "& .MuiSlider-valueLabel": {
        lineHeight: 1.2,
        fontWeight: "bold",
        fontSize: "15px",
        background: "unset",
        color: "black",
        padding: 0,
        width: 32,
        height: 32,
        borderRadius: "50% 50% 50% 0",
        backgroundColor: "#e4e0da",
        transformOrigin: "bottom left",
        transform: "translate(50%, -100%) rotate(-45deg) scale(0)",
        "&:before": { display: "none" },
        "&.MuiSlider-valueLabelOpen": {
            transform: "translate(50%, -100%) rotate(-45deg) scale(1)",
        },
        "& > *": {
            transform: "rotate(45deg)",
        },
    },
});

const Title = ({ text }) => {
    return (
        <Typography
            style={{ fontWeight: "bold", color: "#2a702d" }}
            component="h1"
            variant="h4"
        >
            <br />
            {text} <br />
        </Typography>
    );
};

export { Logos, StyledBox, StyledButton, StyledSlider, Title };
