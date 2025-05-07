// Current state
vec2 globalFragCoord;
float seed; 

// Lights
vec3 ambientLight = cream * .9;
const int TOTAL_LIGHTS = 2;
DirectionalLight lights[TOTAL_LIGHTS];

void setupLights()
{
    lights[0] = DirectionalLight( normalize(vec3(1.0, 1.0, -1.)), white);
    lights[1] = DirectionalLight( normalize(vec3(-1.0, 0.0, -1.)), white);
}

// Materials 
Material objectMaterials[TOTAL_OBJECTS];



void setUpMaterials()
{
/*
    struct Material
    {
        float roughness; 
        vec3  albedo;
        int material_type; 
        float refract_idx; 
    };
*/
    
    objectMaterials[SKY]        = Material(
                                    0.0,
                                    light_blue * .1, 
                                    DIELECTRIC, 
                                    REFRACTIVE_INDEX_AIR);

    objectMaterials[BOUNCE_ORB] = Material(
                                  1.0,
                                  cream ,
                                  DIELECTRIC,
                                  REFRACTIVE_INDEX_FLINT_GLASS);
                                  
    objectMaterials[CENTER_ORB] = Material(
                                  0.0, 
                                  blue,
                                  DIELECTRIC, 
                                  REFRACTIVE_INDEX_FLINT_GLASS);                                

    objectMaterials[FLOOR]       = Material(0.01, 
                                  shaded_blue,
                                  METAL, 
                                  REFRACTIVE_NONE);
    
    objectMaterials[WALL]       = Material(.2, 
                                  near_black,
                                  LAMBERTIAN, 
                                  REFRACTIVE_NONE); 

    objectMaterials[WATER]      = Material(1.0,
                                  navy_blue ,
                                  DIELECTRIC,           
                                  REFRACTIVE_INDEX_WATER); 
  
}

Material getMaterial(int ID)
{
    return objectMaterials[ID];
}

// Bounce animation functions 

float getBounceHeight(float time)
{
    return  cos(time) *16.5; 
}

vec3 getBounceTranslation(float time)
{
    vec3 elipseTranslation = vec3( BOUNCE_RADIUS * sin(time/8.0), getBounceHeight(time), BOUNCE_RADIUS * -1. * cos(time/8.0) - 10.0) ;      
    return elipseTranslation;
}

// Wave animation functions

float getTime() 
{
    // mitigates error over time
    return mod(iTime , 100.);
}

float waveHeight(vec2 pointHorizontal, vec2 wave_source, float speed, float wavelength, float timePhaseOffset, float amplitude) 
{
    // inspired by https://www.shadertoy.com/view/XdtcD4

    float horizontal_dst = distance(pointHorizontal, wave_source); //2d distance to emitter
    
    float phase_shift = (getTime()  + timePhaseOffset) * speed;
    
    // as horizontal_dist increases, sin(x) will oscilate it between troughs and peaks
    // phase shift, shifts the sinusoid creating illusion of ripples
    float wave_height = cos(horizontal_dst * wavelength - phase_shift);
    
    float slope_derivative_factor = EULER;
    float sharper_peaks = pow(slope_derivative_factor, wave_height); // also makes it all positive
    wave_height = sharper_peaks;
    
    wave_height *= amplitude;

	return wave_height;
}

float waveFilter(float dist, float filter_width, float phase_offset, float wavelength)
{
        
    float width_scale = 1.0 / (filter_width / 2.0);
    float start_from_filter_edge = 1.2;
    return max(-pow( (dist ) * width_scale -  phase_offset + start_from_filter_edge, 2.) + 1., 0.);

}

float wavePropogation(vec2 pointHorizontal, vec2  wave_source, float speed, float wavelength, float amplitude, float timeStart) 
{
    // multiply wave function with filter dependent on time to mimic wave propgation
    float waves_height  = waveHeight(pointHorizontal, 
                                  wave_source, 
                                  speed, 
                                  wavelength, 
                                  getTime(),
                                  amplitude) ;

    float horizontal_dst = distance(pointHorizontal, wave_source); //2d distance to emitter
    float phase_offset = (iTime - timeStart) - .2;
    
    
    return waves_height * waveFilter( horizontal_dst,  WAVE_PROP_TIME_FILTER, phase_offset, wavelength);
}

// Object SDF descriptions

float wavesSDF(vec3 p)
{
    float combined_waves_height = 0.;
    vec3 wave_emitter_location;
    
    float half_wavelength = PI; 
    float latest_surface_pierce = half_wavelength * floor(iTime / half_wavelength) ; 

    
    float surfrace_drag_offset;
    float pierce_time = latest_surface_pierce; 

    float center_time = latest_surface_pierce;
    for ( int surface_breach_i = 0; surface_breach_i <  4 ; surface_breach_i++) 
    {
        // bounce orb waves
        for( surfrace_drag_offset = 0.0;  surfrace_drag_offset < 1.0 ; surfrace_drag_offset += .5)
        {
        
            wave_emitter_location.y = getBounceHeight( pierce_time - surfrace_drag_offset);
            wave_emitter_location.xz = getBounceTranslation( (pierce_time + PI/2.0) - surfrace_drag_offset ).xz;

     
            combined_waves_height += wavePropogation(p.xz, 
                                                wave_emitter_location.xz , 
                                                WAVE_SPEED, 
                                                WAVE_LENGTH, 
                                                WAVE_AMPLITUDE_FACTOR, 
                                                pierce_time - surfrace_drag_offset ) ;                                    
        }

        //center orb wave
        combined_waves_height += wavePropogation(p.xz, 
                                                vec2(0.0,-10.0), 
                                                WAVE_SPEED, 
                                                WAVE_LENGTH, 
                                                WAVE_AMPLITUDE_FACTOR, 
                                                pierce_time );  

        pierce_time -= PI; 
        
    }
    
    
    vec3 current_pos    = getBounceTranslation( iTime );

    float waves_only    = p.y - combined_waves_height;
    float centerWater   = sphereSDF(p , 90.0) ;
  //  float followed_body = rectangleSDF(p - vec3(current_pos.x, 11.0, current_pos.z - 15.), vec3( 40. , 50. , 60.)); 
//    return intersectSDF( waves_only, smoothUnion(centerWater, followed_body, 5.));
    return intersectSDF( waves_only, centerWater);


}

float orbSDF(vec3 samplePoint) 
{
    float scale = 14.0;
    vec3 translation =  getBounceTranslation(iTime); 

    float orbDist = sphereSDF((samplePoint - translation), scale) ;
    return orbDist;
}

float centerOrbSDF(vec3 samplePoint)
{
     float scale = 10.0  + max( .3 ,  (1. - abs(cos(iTime))) ) * 5.;
     
     vec3 translation =  vec3(0.0,12.0,-10.0);
     float centerOrb = sphereSDF((samplePoint - translation), scale) ;

     translation =  vec3(cos(iTime) * 24. ,13.0 + sin(iTime / 2.0) * 4.0 , -10.0 + cos(iTime/2.0) * 3.0 );     
     scale = 6.667 +  (1. - abs(cos(iTime))) * 5.;
     float orbit1 = sphereSDF((samplePoint - translation), scale);

     centerOrb = smoothUnion(orbit1, centerOrb, 8.0);

    return centerOrb; 
}

float floorSDF(vec3 samplePoint)
{
    vec3 boxScale = vec3(300.0,2.0,300.0) * 1.;
    vec3 box_translation = vec3(0.0,-120.0,0.0);
    float rect = rectangleSDF(samplePoint - box_translation, boxScale);
    
    vec3 pillarScale = vec3(70.0,100.0,10.0) * 1.;
    vec3 pillarTranslation = vec3(120.0,0.0,-175.0) * 1.;
    
    mat3 rotation = rotationMatrix(-PI / 7.0 + -PI / 7.0 * iTime *.25 , 0.0, 0.0);
    float pillar = rectangleSDF(samplePoint - pillarTranslation, pillarScale, rotation);

    return unionSDF(pillar, rect);
}

float wallSDF(vec3 samplePoint)
{
    vec3 translation = vec3(-70.0, 40.0,-200.0 ) ;
    return sphereSDF((samplePoint - translation), 70. );
}

float skyboxSDF(vec3 samplePoint)
{
    return sphereSDF(samplePoint, MAX_DIST);
}

// Scene SDF functions
void carveOverlaps(inout SceneObjectsDistances currentDistances, int carvedObject)
{
    // Each object (excluding the sky) carves into the target SDF, removig any overlapping volume and redfine its edge there.
    // Start at index 1 to ignore sky
    for(int i = 1 ; i < TOTAL_OBJECTS; i++) {
        if(i != carvedObject) {
            currentDistances.data[carvedObject] = differenceSDF(currentDistances.data[carvedObject], currentDistances.data[i]); 
        }
    }
}

SceneObjectsDistances getSDFdistances(vec3 samplePoint)
{ 
    // Get distances for each object. Negative distances means internal 
    
    SceneObjectsDistances objectDistances; 
    
    objectDistances.data[SKY]        = skyboxSDF(samplePoint);
    objectDistances.data[BOUNCE_ORB] = orbSDF(samplePoint);
    objectDistances.data[CENTER_ORB] = centerOrbSDF(samplePoint);
    objectDistances.data[FLOOR]      = floorSDF(samplePoint);
    objectDistances.data[WALL]       = wallSDF( samplePoint);
    objectDistances.data[WATER]      = wavesSDF(samplePoint);
    
    carveOverlaps(objectDistances, WATER);

    return objectDistances;
}

int getCurrentMedium(vec3 samplePoint)
{
    /* Gets the closest surface which object is inside of, including the sky. 
       Sky box extends past everything, so it only returns sky if it is inside nothing else */
       
    SceneObjectsDistances distances = getSDFdistances(samplePoint);
    
    int closestObject = -1;
    float closestDistance = abs(MAX_DIST);
    for(int object_i = 0 ; object_i < TOTAL_OBJECTS; object_i++) {
        float currentDistance = distances.data[object_i];
        if(currentDistance < 0.0 && abs(currentDistance) < closestDistance) {
            closestDistance = abs(currentDistance);
            closestObject = object_i;
        }
    }
    return closestObject;
}

float objectSDF(vec3 samplePoint, int object_id)
{
    //object ID must be valid index >=0 and < TOTAL_OBJECTS
    SceneObjectsDistances distances = getSDFdistances(samplePoint);
    return  distances.data[object_id];
}

vec3 estimateNormalSDF(vec3 point, int object_id)
{    
    // TODO check if sky , if so not valid

    // tetrahedron technique , https://iquilezles.org/articles/normalsSDF
    vec3 normal = normalize( vec3( 1,-1,-1)*objectSDF( point + vec3( 1,-1,-1)*EPSILON , object_id) + 
                      vec3(-1,-1, 1)*objectSDF( point + vec3(-1,-1, 1)*EPSILON , object_id) + 
                      vec3(-1, 1,-1)*objectSDF( point + vec3(-1, 1,-1)*EPSILON , object_id) + 
                      vec3( 1 ,1, 1)*objectSDF( point + vec3( 1 ,1, 1)*EPSILON, object_id) );
                      
    return normal;
}


// Ray-tracing and Ray-marching
/*
    Incident ray =  light to surface
    View ray =  surface to viewer

*/


SDFDescription getClosestSurface(vec3 samplePoint, int sourceMedium)
{
    /* Get distance from closest SDF in entire scene, not including sky */
    
    SceneObjectsDistances distances = getSDFdistances(samplePoint);
    
    SDFDescription closest;
    closest.dist = MAX_DIST;
    closest.objectID = -1;
    closest.normal = vec3(0.0);

    for (int i = 0; i < TOTAL_OBJECTS; ++i) {
        if (i == SKY) continue;
        if (abs(distances.data[i])  < abs(closest.dist)) {
            closest.dist = distances.data[i];
            closest.objectID = i;
        }
    }
    
    if(closest.objectID != -1) {
        closest.normal = estimateNormalSDF(samplePoint, closest.objectID);
        
        // if internal reflection, flip normal
        if(closest.objectID == sourceMedium) {
            closest.normal *= -1.;
        }
    }

    return closest;
}

RayMarchOutput raymarchObjects(vec3 sourcePoint, vec3 rayDir, int sourceMedium, int maxSteps)
{
    /* 
       Ray march until you hit an object or reach max steps 
       Ray will miss mo object hit or if it hits skybox (due to getClosestSurface() ignoring it)
    */
    RayMarchOutput rayHit;
    rayHit.hit = false;// no objects hit by default
    rayHit.hitpoint = sourcePoint;
    rayHit.sdf = getClosestSurface(sourcePoint, sourceMedium);

    float march_depth = MIN_DIST; 
    for(int i = 0; i < maxSteps; i++) {    
        
        // No hit
        if(march_depth >= MAX_DIST) {
            break;
        }
          
        rayHit.hitpoint = sourcePoint + rayDir * march_depth; 
        rayHit.sdf = getClosestSurface(rayHit.hitpoint, sourceMedium);
        
        if(abs(rayHit.sdf.dist) < EPSILON) {
            rayHit.hit = true; 
            break;
        }
        
        // update for next march 
        march_depth += abs(rayHit.sdf.dist);
    }
    
    return rayHit;
}


float fresnelReflectivity(float refraction_ratio, float cos_theta)
{
    // Use Schlick's Fresnel  approximation for reflectance.
    float r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio);
    r0 = r0*r0;
    float reflectance = (r0 * r0)  + (1.0 - r0) * pow((1.0 - cos_theta), 5.0);
    return reflectance; 
}

float lambertCosine(vec3 normal, vec3 pointToLightDir)
{
     return max(dot(normal, pointToLightDir), 0.0);
}

vec3 metalBRDF(Material hitMaterial, vec3 incidentRay, vec3 viewRay, vec3 normal) 
{
    // Simplified - assumes ideal refelctors - describes the color and intensity fo reflected light

    vec3 perfectReflection = reflect(-incidentRay, normal);
    float lambertCos = lambertCosine(viewRay, perfectReflection);
    
    // Fresnel-Schlick approximation for metal reflectivity
    float F0 = hitMaterial.albedo.x; 
    float fresnel = F0 + (1.0 - F0) * pow(1.0 - lambertCos, 5.0);
    
    // Roughness term simpilied
    float shininess = 1.0 / max(hitMaterial.roughness * hitMaterial.roughness, 0.001); // avoid divide by zero
    float specular = pow(lambertCos, shininess);
    
    vec3 bdrf = hitMaterial.albedo * fresnel * specular;
    
    // Simplification - not necessially physically accurate, but gets nice effect
    for(int i = 0; i < TOTAL_LIGHTS; i++){
        float lightCos = lambertCosine(-incidentRay, lights[0].direction);  
        float lightSpecular = pow(lightCos, shininess);
        bdrf += lights[0].radiance / 2. * lightCos ;
    }
    

    // Energy conservation: modulate by Fresnel and albedo
    return bdrf;
    
}

ScatteredRay scatterMetal(vec3 hitpoint, vec3 incidentRay, vec3 normal, int sourceMedium, int hitObject)
{
    ScatteredRay scatteredRay;
    Material hitMaterial = getMaterial(hitObject);

    // Importance sampling around perfect mirror direction.
    vec3 randVec  = sampleUnitSphere(seed);
    vec3 reflected = normalize(reflect(incidentRay, normal) + hitMaterial.roughness * randVec);
    bool away_from_surface = dot(reflected, normal) > 0.0; // if scattered direction is in same hemisphere as normal
    
    if(away_from_surface) {
        scatteredRay.ray.rayDir = reflected;
        scatteredRay.energy = metalBRDF(hitMaterial, -reflected, incidentRay,  normal);
    }
    else {            
        // if not scattered, it is absorbed into object; No scatter direction - end raytracing here
        scatteredRay.ray.rayDir = vec3(0.0);
        scatteredRay.energy = vec3(0.0);
    }
    
    scatteredRay.ray.origin = hitpoint + scatteredRay.ray.rayDir * EPSILON;
    
    return scatteredRay;
}

vec3 dieletricSampleContribution(bool reflected, float probability, vec3 albedo) 
{
    /*
        Using monte carlo sampling, BDRF ends up being 1 (either it reflects or refracts), 
        but to keep the probailistic expected value we weight the sampled by the probaility to ensure unbiased sampling
    */
    if(reflected){
        return vec3(1.0 / probability); 
    } else { // refracted
        return albedo / vec3( 1.0 - probability);
    }
}

ScatteredRay scatterDieletric(vec3 hitpoint, vec3 incidentRay, vec3 normal, int sourceMedium, int hitObject)
{
    ScatteredRay scatteredRay;
    float enter_idx = getMaterial(sourceMedium).refract_idx;
    float exit_idx  = getMaterial(hitObject).refract_idx; 
    float refraction_ratio = enter_idx / exit_idx;

    //using snells law we can find when the equation has no solution:  sin(theta) = refration_ratio * sin'(theta') when right hand side > 1 (max of sin is 1) 
    float cos_theta = min(dot(-incidentRay, normal), 1.0); // calculcate sign by using cosign ans dot product properties
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    bool cannot_refract = refraction_ratio * sin_theta > 1.0;  // if cant reflect due to snells law
    
    // Reflect randomly based on fraction of light energy reflected - over time this statistically reproduces energy split
    float percentReflected = fresnelReflectivity(refraction_ratio, cos_theta);  
   
   // float seed = noiseFromVec2(vec2(gl_FragCoord.xy / iResolution.xy) * float(bounceDephtCopy) + iTime);
    bool reflected = cannot_refract || percentReflected >  randomNoise(9.);
    if (reflected) {
        scatteredRay.ray.rayDir = reflect(incidentRay, normal);
    }
    else {
        scatteredRay.ray.rayDir = refract(incidentRay, normal, refraction_ratio);  
    }
    
    vec3 targetAlbedo =  getMaterial(hitObject).albedo;
    scatteredRay.energy = dieletricSampleContribution(reflected, percentReflected, targetAlbedo);
    scatteredRay.ray.origin = hitpoint + scatteredRay.ray.rayDir * EPSILON;
    
    return scatteredRay;

}

vec3 lambertianSampleContribution(vec3 albedo)
{
    /* 
        The rendering equation is: outgoing_radiance = BDRF * incomingRadaince * cos(theta)
        In monte carlo, we weight our outgoing light by probability to unbiased sampling : outgoing_radiance / probaility 
        
        so we can simply 
        BRDF =  albedo / PI
        probablity = cos(theta) / PI 
        
        so outgoing radiance =   ( albedo / PI  * incomingRadaince * cos(thetha) ) /  (cos(theta) / PI) 
        clearing cancel out terms = albedo * incomingRadiance

        so sample contribution = albedo
    */ 
    return albedo;
}

ScatteredRay scatterLambertian(vec3 hitpoint, vec3 incidentRay, vec3 normal, int sourceMedium, int hitObject)
{
    ScatteredRay scatteredRay;
    Material hitMaterial = getMaterial(hitObject);
    
    vec3 randVec = cosineSampleHemisphere(normal, seed);
    vec3 scatter_direction = normalize(normal + randVec * hitMaterial.roughness);
    
    // Catch degenerate scatter direction, avoids error where diffuse bounce collapsing back to the point (zero direction) 
    if (near_zero(scatter_direction)) {
        scatter_direction = normal;
    }
    
    scatteredRay.ray.rayDir = scatter_direction;
    scatteredRay.ray.origin = hitpoint + scatteredRay.ray.rayDir * EPSILON;
    scatteredRay.energy = lambertianSampleContribution(hitMaterial.albedo);     // Update energy left for incoming indirect light
    
    return scatteredRay;

}

bool inShadow(vec3 sourcePoint, int sourceMedium, vec3 lightDir) 
{
    
    const int MAX_SHADOW_BOUNCES = 4;
    bool shadowed = false;
    Ray shadowRay;
    shadowRay.rayDir = -lightDir;
    shadowRay.origin = sourcePoint +  shadowRay.rayDir * EPSILON*2.0;


    for (int i = 0; i < MAX_SHADOW_BOUNCES; ++i) {
        int medium = getCurrentMedium(shadowRay.origin + shadowRay.rayDir * EPSILON*2.0);

        RayMarchOutput intersection = raymarchObjects(shadowRay.origin, shadowRay.rayDir, medium, 50);

        if (!intersection.hit || (intersection.hit && intersection.sdf.objectID == SKY) ) {  // Reached sky
            shadowed = false;
            break;
        }
        
        int hitObject = intersection.sdf.objectID;
        Material mat = getMaterial(hitObject);

        if (mat.material_type == DIELECTRIC) {
            // Refract through object
            shadowRay = scatterDieletric(intersection.hitpoint, shadowRay.rayDir, intersection.sdf.normal, medium, hitObject).ray;
        
        } else {
            shadowed = false;
        }
    }

    return shadowed; // Too many bounces = consider shadowed
}

vec3 lambertianDirectLightRadiance(vec3 hitpoint, vec3 viewRay, vec3 normal, int sourceMedium, int hitObject)
{
    Material hitMaterial = getMaterial(hitObject);

    // Get how much light survives the interaction
    vec3 directLightRadiance = vec3(0.);
    for(int i = 0; i < TOTAL_LIGHTS; i++) {
        
        vec3 lightDir = lights[i].direction;
        //bool visible = !inShadow(hitpoint, sourceMedium, lightDir); 
        vec3 bdrf = hitMaterial.albedo / PI; 

        directLightRadiance += bdrf * lights[i].radiance * lambertCosine(normal, lightDir) ; // encodes L_o = BDRF * L_in * cos(thetha)
    }
    return directLightRadiance;
}
 


ScatteredRay getOutgoingRadiance(vec3 hitpoint, vec3 viewRay, vec3 normal, int sourceMedium, int hitObject) 
{
    
    ScatteredRay scatteredRay;
    Material hitMaterial = getMaterial(hitObject);
    Material sourceMaterial = getMaterial(sourceMedium);

    if( hitMaterial.material_type == DIELECTRIC) {
        scatteredRay = scatterDieletric(hitpoint, viewRay, normal, sourceMedium, hitObject); 
    }  
    else if(hitMaterial.material_type == LAMBERTIAN) { // lambertian refelction
        scatteredRay = scatterLambertian(hitpoint, viewRay, normal, sourceMedium, hitObject); // indirect lighting   
    } 
    else if (hitMaterial.material_type == METAL) {
        scatteredRay = scatterMetal(hitpoint, viewRay, normal, sourceMedium, hitObject); 
    }

    return scatteredRay;
}

vec3 getEnvironmentalLight(vec3 rayDirection) {
    
    vec3 radiance = vec3(0.0); 
    // Get radiance from directional light if it aligns
    for(int i = 0; i < TOTAL_LIGHTS; i++) {
        DirectionalLight light = lights[i];
        float lightHit = dot(rayDirection, light.direction);
        if(lightHit > 1. - EPSILON ) {
            radiance += lights[i].radiance;
        }
        //radiance += lights[i].radiance * lambertCosine(rayDirection, light.direction); 
        
    }
    radiance += ambientLight;
    return radiance;   
}

vec3 rayTrace(vec3 rayOrigin, vec3 rayDirWorld)
{
    Ray currentRay; 
    currentRay.origin = rayOrigin; 
    currentRay.rayDir = rayDirWorld; 
    
    vec3 color = vec3(0.0,0.0,0.0);
    vec3 energy = vec3(1.0f);
    bool absorbed = false;
    for(int current_bounce_depth = 0; current_bounce_depth < MAX_BOUNCES; current_bounce_depth++)
    {
        int medium = getCurrentMedium(currentRay.origin + currentRay.rayDir * EPSILON*2.0);

        RayMarchOutput intersection  = raymarchObjects(currentRay.origin, currentRay.rayDir, medium, MAX_RAYMARCH_STEPS);

        if(!intersection.hit) {
            //color += getEnvironmentalLight(currentRay.rayDir) * energy; 
            absorbed = false;
            break;
        }
        
        vec3 hitpoint = intersection.hitpoint;
        int  hitObject = intersection.sdf.objectID;
        vec3 normal = intersection.sdf.normal;
        
        // Get how much light (including its color) survives interaction and it's direction
        ScatteredRay radiance = getOutgoingRadiance(hitpoint, currentRay.rayDir, normal, medium, hitObject); 
    
        // Update for next bounce
        currentRay = radiance.ray;  
        energy *= radiance.energy;

         if(SHOW_NORMALS == true) {
             color = normal;
        }

 
        // terminate if ray no longer has valid direction - might be absorbed into material
        if(length(currentRay.rayDir) == 0.0) {
            absorbed = true;
            break;
        }
    }
    
    if(!absorbed) {
        color += getEnvironmentalLight(currentRay.rayDir) * energy; 
    }
    return color;
}

vec3 updateEyePosition(vec3 eyePosition, vec4 mouseNDC)
{    
   float zoomScale = iMouse.y > 0.0 ? -mouseNDC.y : 0.0; 
   eyePosition.z += (zoomScale < 0.70) ? zoomScale * 200.0 : zoomScale *  400. ;
   eyePosition.y += (zoomScale < 0.70) ? zoomScale * 100.0 : zoomScale *  800. ;
  
   float yawScale =  iMouse.y > 0.0 ? mouseNDC.x : 0.0;
   float yawAngle = yawScale * (PI / 1.5);
   eyePosition = rotationMatrix(yawAngle, 0., 0.) * eyePosition;
    
   return eyePosition;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Intialize arrays
    setUpMaterials();
    setupLights(); 

    globalFragCoord = fragCoord;

    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = globalFragCoord/iResolution.xy;
    vec4 mouseNorm = iMouse/ iResolution.xyxy;
    vec4 mouseNDC = (mouseNorm - .5 ) * 2.0;
    
    
    // random seed inspired by https://www.shadertoy.com/view/fdS3zw
    vec2 p = -1.0 + 2.0 * (fragCoord.xy) / iResolution.xy;
    p.x *= iResolution.x/iResolution.y;
    seed = p.x + p.y * 3.43121412313 + fract(1.12345314312);

    // eye will serve as ray origin
    vec3 eyePosition = vec3(0.0, 0.0, 320.0);
    eyePosition = updateEyePosition(eyePosition, mouseNDC);

    
    
    // Tranforms from camera space to world space
    mat4 viewToWorld = viewMatrix(eyePosition, vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    vec3 viewDir;
    vec3 rayDirWorld;
    
    // ray tracing
    vec3 final_color = vec3(0.0); 
    vec2 target_pixel;
            
    for(int sample_i = 0; sample_i < MAX_SAMPLES; sample_i++)
    {
        //anti aliaising get samples near target smaple and blends them
        float r = randomNoise(dot(fragCoord, vec2(12.9898, 78.233)) + float(sample_i));
        float theta = 2.0 * PI * r;
        vec2 antiAlias_offset = vec2(cos(theta), sin(theta)) * MAX_ANTIALIAS_OFFSET * randomNoise(iTime + float(sample_i)) * MAX_ANTIALIAS_OFFSET;
        target_pixel = fragCoord  + antiAlias_offset;
        
        //  Ray trace
        viewDir     =  rayDirection(45.0, iResolution.xy, target_pixel); 
        rayDirWorld = (viewToWorld * vec4(viewDir, 0.0)).xyz;
        final_color += rayTrace(eyePosition, rayDirWorld) ;
    }
    
    if(MAX_SAMPLES > 0) {
        final_color /= float(MAX_SAMPLES);
    }
    
    fragColor = vec4(pow(final_color, vec3(1.0/2.2)), 1.0);  // gamma correction

}