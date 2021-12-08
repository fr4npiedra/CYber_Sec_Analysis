<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

use App\Models\Shootings;
use App\Models\Cyberanalysis;
use App\Models\CyberRecord;
use App\Models\CyberTest;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| Here is where you can register API routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| is assigned the "api" middleware group. Enjoy building your API!
|
*/

Route::middleware('auth:sanctum')->get('/user', function (Request $request) {
    return $request->user();
});


Route::get('/shootings', function() {

	$newsData = json_decode(Shootings::all());

	return response()->json($newsData);

});

Route::get('/cyberrecords', function() {

	$newsData = json_decode(CyberRecord::all());

	return response()->json($newsData);

});

Route::get('/cybertests', function() {

	$newsData = json_decode(CyberTest::all());

	return response()->json($newsData);

});