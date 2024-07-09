import { getResult } from "./forward.js";
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import cors from "cors"
import opn from "opn"
const PORT = 3000;


const app = express();
app.use(cors())
app.use(express.json());
app.use(express.static('./public'));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, './index.html'));
});

app.post('/pridict', async (req, res) => {
    let data = req.body;
    let pr = getResult(data.data)
    let response = { pridiction: pr }
    res.json(response);
})

app.listen(PORT, () => {
    console.log('server running at localhost:' + PORT);
    opn('http://localhost:' + PORT); 
});
