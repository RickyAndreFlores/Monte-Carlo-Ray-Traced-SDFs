
// Toggle
const bool SHOW_NORMALS = false;

// Sampling and bounce control
const int MAX_BOUNCES = 6;
const int MAX_SAMPLES = 5;
const float MAX_ANTIALIAS_OFFSET = 1.f;

// Raymarching values
const int   MAX_RAYMARCH_STEPS = 85;
const float MIN_DIST = 0.01; 
const float MAX_DIST = 1500.0;
const float EPSILON  = 0.0001;

// Refractive indices
const float REFRACTIVE_INDEX_AIR = 1.0003;
const float REFRACTIVE_INDEX_GLASS = 1.125;
const float REFRACTIVE_INDEX_WATER = 1.333;
const float REFRACTIVE_INDEX_FLINT_GLASS = 1.655;
const float REFRACTIVE_NONE = -1.0;

// Math constants
const float PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;
const float EULER = 2.7182818284590452353602874;

// Wave simulation
const float WAVE_SPEED = 9.02;
const float WAVE_LENGTH = 0.5;
const float WAVE_AMPLITUDE_FACTOR = 0.5;
const float WAVE_PROP_TIME_FILTER = 50.0;
const float BOUNCE_RADIUS = 45.0;

// Scene Objects 
const int TOTAL_OBJECTS = 6;
// Object IDs
const int SKY = 0;
const int CENTER_ORB = 1;
const int BOUNCE_ORB = 2;
const int FLOOR = 3;
const int WALL = 4;
const int WATER = 5;

// Material types
const int LAMBERTIAN = 0;
const int METAL = 1;
const int DIELECTRIC = 2;

// Colors
const vec3 light_blue  = vec3(0.27,0.73,.86) ;
const vec3 near_black  = vec3(0.4);
const vec3 navy_blue   = vec3(75., 78., 109.) / 256. ;
const vec3 muddy_green = vec3(115., 104., 59.) / 256. ;
const vec3 yellow      = vec3(235., 252., 27.) / 256. ; 
const vec3 shaded_blue = vec3(78, 107, 167) / 256. ;
const vec3 white       = vec3(1.0);
const vec3 cream       = vec3(255.0, 231.0, 179.0) / 256.;
const vec3 blue        = vec3(59., 151., 217.) / 256.;
const vec3 teal        = vec3(68,255,209) / 256.;

// Structs
struct Material
{
    float roughness; 
    vec3  albedo;
    int material_type; 
    float refract_idx; 
    
};

struct SDFDescription
{
    // Distance to object of objectID
    float dist;
    int objectID;
    vec3 normal;
};

struct SceneObjectsDistances
{
    float data[TOTAL_OBJECTS]; 
};

struct Ray
{
    vec3 origin; 
    vec3 rayDir;
};

struct ScatteredRay 
{
    Ray ray; 
    vec3 energy;
};

struct DirectionalLight {
    vec3 direction; // direction towards light 
    vec3 radiance;
};

struct RayMarchOutput {
    bool hit;
    vec3 hitpoint;
    SDFDescription sdf;
};

struct RayTraceOutput
{
    vec3 hit_color;
    vec3 rayDir; 
};


// SDF functions

float sphereSDF(vec3 point, float radius)
{    
    // Sphere centered on the origin
    return length(point) - radius;
}

float rectangleSDF(vec3 point, vec3 halfExtent, mat3 rotationMatrix)
{
    point = transpose(rotationMatrix) * point;

    // Rectangle centered at origin, with a the given half extent 
    vec3 q = abs(point) - halfExtent; // move p to positive quadrant, then subtract from the box's extent'
    float externalDistanceToSurface = length(max(q,0.0)); // 0 if point is inside box
    float shallowestPenetrationDepth = min(max(q.x,max(q.y,q.z)),0.0); // 0 if point lies outside the box
    return externalDistanceToSurface + shallowestPenetrationDepth; // externalDistance or internalDistance
}


float rectangleSDF(vec3 point, vec3 halfExtent)
{
    // Rectangle centered at origin, with a the given half extent 
    vec3 q = abs(point) - halfExtent; // move p to positive quadrant, then subtract from the box's extent'
    float externalDistanceToSurface = length(max(q,0.0)); // 0 if point is inside box
    float shallowestPenetrationDepth = min(max(q.x,max(q.y,q.z)),0.0); // 0 if point lies outside the box
    return externalDistanceToSurface + shallowestPenetrationDepth; // externalDistance or internalDistance
}

float smoothUnion(float a, float b, float k)
{
    // credit https://iquilezles.org/articles/smin
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*h*k*(1.0/6.0);
}

float unionSDF(float distA, float distB)
{
    return min(distA, distB);
}

// Carve out object B from Object A 
float differenceSDF(float distA, float distB) {
    return max(distA, -distB);
}

float smoothDifference(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d1 + d2) / k, 0.0, 1.0);
    return mix(d1, -d2, h) + k * h * (1.0 - h);
}

float intersectSDF(float distA, float distB) 
{
    return max(distA, distB);
}

// Noise functions

float randomNoise(float seed) { return fract(sin(seed)*43758.5453123); }

float noiseFromVec2(vec2 vec) { return fract(sin(dot(vec, vec2(12.9898,78.233))) * 43758.5453); }

// Credit : https://www.shadertoy.com/view/fdS3zw
vec2 hash2(inout float seed) {
    return fract(sin(vec2(seed+=0.1,seed+=0.1))*vec2(43758.5453123,22578.1459123));
}
// Credit : https://www.shadertoy.com/view/fdS3zw
vec3 cosineSampleHemisphere(vec3 n, inout float seed)
{
    vec2 u = hash2(seed);

    float r = sqrt(u.x);
    float theta = 2.0 * PI * u.y;
 
    vec3  B = normalize( cross( n, vec3(0.0,1.0,1.0) ) );
	vec3  T = cross( B, n );
    
    return normalize(r * sin(theta) * B + sqrt(1.0 - u.x) * n + r * cos(theta) * T);
}

vec3 sampleUnitSphere(inout float seed)
{
    // Generate random values for phi and theta
    vec2 u = hash2(seed);

    // theta is in the range [0, 2 * pi]
    float theta = 2.0 * PI * u.x;

    // phi is in the range [0, pi]
    float phi = acos(2.0 * u.y - 1.0);  // This ensures uniform distribution over the surface

    // Convert spherical coordinates to Cartesian coordinates
    float sin_phi = sin(phi);
    vec3 sampledPoint = vec3(
        sin_phi * cos(theta),
        sin_phi * sin(theta),
        cos(phi)
    );

    return normalize(sampledPoint);
}


// Ray tracing utility

bool near_zero(vec3 vec)
{
     // Return true if the vector is close to zero in all dimensions.
     // this is useful in avoiding float point error leading to a near zero value when it should just be zero
    return (abs(vec.x) < EPSILON) && (abs(vec.y) < EPSILON) && (abs(vec.z) < EPSILON);
}

vec3 randomInsideUnitSphere(vec2 seed) {
    while (true) {
        vec3 p = vec3(
            noiseFromVec2(seed),  // replace with your random [-1, 1] function
            noiseFromVec2(seed),
            noiseFromVec2(seed)
        ) * 2.0 - 1.0;

        float lenSq = dot(p, p);
        if (lenSq > 1e-8 && lenSq <= 1.0) {
            return normalize(p / sqrt(lenSq));
        }
    }
}


vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    /**
     * Return the normalized direction to march in from the eye point for a single pixel.
     * 
     * fieldOfView: vertical field of view in degrees
     * size: resolution of the output image
     * fragCoord: the x,y coordinate of the pixel in the output image
     */

    vec2 xy = fragCoord - size / 2.0; // centers 'screen' at origin
    // height from origin, angle is half of field of view
    float z = (size.y  / tan(radians(fieldOfView) / 2.0) ); // adj = oppiste / tan()
    
    // normalize to make unit vecotr, -z = camera pointing in negative z direction
    vec3 vector_to_pixel = vec3(xy.x, 0, 0) +  vec3(0, xy.y, 0) + vec3(0, 0, -z);
    return normalize(vector_to_pixel); // make unit vector
}

// Matricies utility 

mat4 viewMatrix(vec3 eye, vec3 center, vec3 up)
{
    // based off jaime-wongs tutorial http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
    vec3 forward     = normalize(center - eye); 
    vec3 side        = normalize(cross(forward, up)); 
    vec3 relative_up = cross(side, forward);

    return mat4(
           vec4(side, 0.0), 
           vec4(relative_up, 0.0), 
           vec4(-forward, 0.0),
           vec4(0.0,0.0,0.0,1.0)
        );
}

mat3 rotationMatrix(float yaw, float pitch, float roll) {
    // Yaw (Y-axis)
    float cy = cos(yaw);
    float sy = sin(yaw);
    mat3 R_yaw = mat3(
        vec3( cy, 0.0, -sy),
        vec3(0.0, 1.0,  0.0),
        vec3( sy, 0.0,  cy)
    );

    // Pitch (X-axis)
    float cp = cos(pitch);
    float sp = sin(pitch);
    mat3 R_pitch = mat3(
        vec3(1.0,  0.0,  0.0),
        vec3(0.0,  cp,  -sp),
        vec3(0.0,  sp,   cp)
    );

    // Roll (Z-axis)
    float cr = cos(roll);
    float sr = sin(roll);
    mat3 R_roll = mat3(
        vec3( cr, -sr, 0.0),
        vec3( sr,  cr, 0.0),
        vec3(0.0, 0.0, 1.0)
    );

    // Combine: roll → pitch → yaw
    return R_yaw * R_pitch * R_roll;
}

// Misc utility 
float normSin(float x)
{
     return (sin(x)  + 1.0 ) /2.0;
}

float normCos(float x)
{
     return (cos(x)  + 1.0 ) /2.0;
}

vec3 mix3(vec3 firstColor, vec3 middleColor, vec3 endColor,float weight)
{
    float h = 0.8; // adjust position of middleColor
    return mix(mix(firstColor, middleColor, weight/h), mix(middleColor, endColor, (weight - h)/(1.0 - h)), step(h, weight));
}