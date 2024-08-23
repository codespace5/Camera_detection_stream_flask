import './App.css';

function App() {
  return (
    <div className="App">
      <div className='title'>Camera</div>
      <div className='camera'>
        <div className='camera1'>
          <img src={'http://localhost:8000/stream'} className='image'  alt="logo" />
          <div>Camera 1</div>
        </div>
        <div className='camera2'>
            <img src={'http://localhost:8000/stream1'} className='image'  alt="logo" />
            <div>Camera 2</div>
        </div>
        
        
      </div>
    </div>
  );
}

export default App;
