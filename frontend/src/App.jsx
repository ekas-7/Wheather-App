import { useState, useEffect } from 'react';
import { Search, CloudRain, Wind, Droplets, ThermometerSun, Umbrella, ArrowDown, ArrowUp, RefreshCw, Sun, Cloud, CloudLightning, CloudSnow, CloudDrizzle, CloudFog } from 'lucide-react';

// Main Weather App Component
export default function WeatherApp() {
  const [city, setCity] = useState('New York');
  const [weatherData, setWeatherData] = useState(null);
  const [forecast, setForecast] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchInput, setSearchInput] = useState('');

  // Fetch weather data when city changes
  useEffect(() => {
    setLoading(true);
    setError(null);
    
    // Simulate API call
    setTimeout(() => {
      const newData = getMockWeatherData(city);
      if (newData.error) {
        setError(newData.error);
        setWeatherData(null);
      } else {
        setWeatherData(newData.current);
        setForecast(newData.forecast);
      }
      setLoading(false);
    }, 800);
  }, [city]);

  const handleSearch = () => {
    if (searchInput.trim()) {
      setCity(searchInput);
      setSearchInput('');
    }
  };

  const handleRefresh = () => {
    setLoading(true);
    // Simulate API refresh
    setTimeout(() => {
      const newData = getMockWeatherData(city);
      setWeatherData(newData.current);
      setForecast(newData.forecast);
      setLoading(false);
    }, 800);
  };

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 p-4">
      <header className="text-center mb-6">
        <h1 className="text-3xl font-bold text-blue-800 mb-2">Weather Dashboard</h1>
        <div className="max-w-md mx-auto flex items-center">
          <input
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Search for a city..."
            className="flex-grow p-2 rounded-l-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button 
            onClick={handleSearch}
            className="bg-blue-600 text-white p-2 rounded-r-lg hover:bg-blue-700"
          >
            <Search size={20} />
          </button>
        </div>
      </header>

      {loading ? (
        <div className="flex justify-center items-center flex-grow">
          <div className="flex items-center gap-2">
            <RefreshCw className="animate-spin text-blue-600" size={24} />
            <span className="text-gray-600">Loading weather data...</span>
          </div>
        </div>
      ) : error ? (
        <div className="text-center text-red-500 p-4 bg-red-50 rounded-lg">
          {error}
        </div>
      ) : weatherData ? (
        <div className="flex flex-col lg:flex-row gap-6">
          <CurrentWeather 
            data={weatherData} 
            city={city} 
            onRefresh={handleRefresh} 
          />
          <ForecastSection forecast={forecast} />
        </div>
      ) : null}
    </div>
  );
}

// Current Weather Component
function CurrentWeather({ data, city, onRefresh }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 flex-1">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-2xl font-semibold text-gray-800">{city}</h2>
          <p className="text-gray-500">{new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}</p>
        </div>
        <button 
          onClick={onRefresh} 
          className="text-blue-500 hover:text-blue-700"
          aria-label="Refresh weather data"
        >
          <RefreshCw size={20} />
        </button>
      </div>
      
      <div className="flex items-center justify-center my-6">
        <div className="text-center">
          {getWeatherIcon(data.condition, 64)}
          <h3 className="text-5xl font-bold my-4">{data.temperature}°</h3>
          <p className="text-xl text-gray-700">{data.condition}</p>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mt-6">
        <WeatherDetail icon={<ThermometerSun size={20} />} label="Feels Like" value={`${data.feelsLike}°`} />
        <WeatherDetail icon={<Wind size={20} />} label="Wind" value={`${data.windSpeed} km/h`} />
        <WeatherDetail icon={<Droplets size={20} />} label="Humidity" value={`${data.humidity}%`} />
        <WeatherDetail icon={<Umbrella size={20} />} label="Precipitation" value={`${data.precipitation}%`} />
      </div>
      
      <div className="flex justify-between items-center mt-6 text-sm">
        <div className="flex items-center">
          <ArrowDown className="text-blue-500 mr-1" size={16} />
          <span>Low: {data.low}°</span>
        </div>
        <div className="flex items-center">
          <ArrowUp className="text-red-500 mr-1" size={16} />
          <span>High: {data.high}°</span>
        </div>
      </div>
    </div>
  );
}

// Weather Detail Component
function WeatherDetail({ icon, label, value }) {
  return (
    <div className="flex items-center p-3 bg-blue-50 rounded-lg">
      <div className="text-blue-500 mr-3">
        {icon}
      </div>
      <div>
        <p className="text-xs text-gray-500">{label}</p>
        <p className="font-medium">{value}</p>
      </div>
    </div>
  );
}

// Forecast Section Component
function ForecastSection({ forecast }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 flex-1">
      <h3 className="text-xl font-semibold text-gray-800 mb-4">5-Day Forecast</h3>
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        {forecast.map((day, index) => (
          <ForecastCard key={index} day={day} />
        ))}
      </div>
    </div>
  );
}

// Forecast Card Component
function ForecastCard({ day }) {
  return (
    <div className="bg-blue-50 rounded-lg p-3 text-center">
      <p className="font-medium">{day.day}</p>
      <div className="my-2">
        {getWeatherIcon(day.condition, 36)}
      </div>
      <p className="text-sm">{day.condition}</p>
      <div className="flex justify-between items-center mt-2 text-sm">
        <div className="flex items-center">
          <ArrowDown className="text-blue-500" size={12} />
          <span>{day.low}°</span>
        </div>
        <div className="flex items-center">
          <ArrowUp className="text-red-500" size={12} />
          <span>{day.high}°</span>
        </div>
      </div>
    </div>
  );
}

// Weather Icon Helper
function getWeatherIcon(condition, size) {
  const lowerCondition = condition.toLowerCase();
  
  if (lowerCondition.includes('clear') || lowerCondition.includes('sunny')) {
    return <Sun size={size} className="text-yellow-500 mx-auto" />;
  } else if (lowerCondition.includes('rain')) {
    return <CloudRain size={size} className="text-blue-500 mx-auto" />;
  } else if (lowerCondition.includes('cloud')) {
    return <Cloud size={size} className="text-gray-500 mx-auto" />;
  } else if (lowerCondition.includes('thunder') || lowerCondition.includes('storm')) {
    return <CloudLightning size={size} className="text-purple-500 mx-auto" />;
  } else if (lowerCondition.includes('snow')) {
    return <CloudSnow size={size} className="text-blue-200 mx-auto" />;
  } else if (lowerCondition.includes('drizzle')) {
    return <CloudDrizzle size={size} className="text-blue-400 mx-auto" />;
  } else if (lowerCondition.includes('fog') || lowerCondition.includes('mist')) {
    return <CloudFog size={size} className="text-gray-400 mx-auto" />;
  } else {
    return <Cloud size={size} className="text-gray-400 mx-auto" />;
  }
}

// Mock Weather Data Generator
function getMockWeatherData(city) {
  // For demo purposes - in a real app this would be an API call
  const validCities = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Berlin', 'Moscow'];
  const lowercaseCity = city.toLowerCase();
  
  if (!validCities.some(c => c.toLowerCase() === lowercaseCity)) {
    return {
      error: `Weather data for "${city}" is not available. Try New York, London, Tokyo, Paris, Sydney, Berlin, or Moscow.`
    };
  }
  
  const weatherConditions = ['Sunny', 'Partly Cloudy', 'Cloudy', 'Rainy', 'Thunderstorm', 'Snow', 'Foggy'];
  const randomCondition = () => weatherConditions[Math.floor(Math.random() * weatherConditions.length)];
  
  const baseTemp = 10 + Math.floor(Math.random() * 25);
  
  const current = {
    temperature: baseTemp,
    condition: randomCondition(),
    feelsLike: baseTemp - 2 + Math.floor(Math.random() * 4),
    windSpeed: 5 + Math.floor(Math.random() * 20),
    humidity: 30 + Math.floor(Math.random() * 60),
    precipitation: Math.floor(Math.random() * 100),
    low: baseTemp - 5 - Math.floor(Math.random() * 3),
    high: baseTemp + 2 + Math.floor(Math.random() * 5)
  };
  
  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];
  const forecast = days.map(day => {
    const dayTemp = baseTemp - 5 + Math.floor(Math.random() * 10);
    return {
      day,
      condition: randomCondition(),
      low: dayTemp - Math.floor(Math.random() * 5),
      high: dayTemp + Math.floor(Math.random() * 8)
    };
  });
  
  return { current, forecast };
}