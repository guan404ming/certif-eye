import React, { useState } from "react";

import { Marker, Map, type MapMouseEvent } from "@vis.gl/react-google-maps";

function LocationPicker() {
  const [place, setPlace] = useState<MapMouseEvent | null>(null);

  return (
    <>
      <Map
        defaultCenter={{ lat: 25.0129, lng: 121.5371 }}
        defaultZoom={15}
        onClick={(e) => setPlace(e)}
        mapId={process.env.NEXT_PUBLIC_MAP_ID}
        className="w-full h-96"
      >
        <Marker
          position={
            place && place.detail.latLng
              ? {
                  lat: place.detail.latLng.lat,
                  lng: place.detail.latLng.lng,
                }
              : { lat: 25.0129, lng: 121.5371 }
          }
        />
      </Map>
      {place && (<div>{place.detail.placeId}</div>)}
    </>
  );
}

export default React.memo(LocationPicker);
