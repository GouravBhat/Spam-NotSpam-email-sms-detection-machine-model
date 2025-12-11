import { useState } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
 import './App.css'

function App() {
  // const [count, setCount] = useState(0)
  const[message,setMessage]=useState("")
  const[prediction,setPrediction]=useState("")
  const[loading,setLoading]=useState(false)
  const[errormessage,setErrorMessage]=useState("")
  const submit_handler = async (e) => {
  e.preventDefault();
  
   // optional: clear previous result

  try {
    setLoading(true);
    setPrediction("");  
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await res.json();
    if(data.data){
      setPrediction(data.data);
    }
    else{
      setErrorMessage(data.message)
    }
  } catch (error) {
    setPrediction("Error: Server not responding");
  }

  setLoading(false);
  setMessage("")
};

  return (
    <div className="container">
      <form className="card" onSubmit={submit_handler}>
        <h1>Spam / Ham Classifier</h1>

        <textarea
          placeholder="Enter your message..."
          onChange={(e) => setMessage(e.target.value)}
          value={message}
          required
        />

        <button type="submit">Check Message</button>

         {loading && <p className="loading">Checking...</p>}

         {!loading && prediction && (
        <h3 className="result">Prediction: {prediction}</h3>
         )}
         {!loading && !prediction && (
        <h3>{errormessage} </h3>
         )}
        
      </form>
    </div>
  )
}

export default App
